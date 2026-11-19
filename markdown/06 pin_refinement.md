"""Height-only candidate extraction and rectangle refinement AFTER ICP.

No clicking, no marker extraction, no known dimensions, no voxelization of output.
Requirements: numpy, scipy, open3d, opencv-python.
Units: mm. Run on Windows with the ICP outputs from planeICP(1).md.

IMPORTANT: automatic identification is geometric and data-dependent. If detection or
correspondence is ambiguous, the script SKIPS that pair instead of forcing a match.
Height-only filtering will NOT distinguish markers from pins. Check diagnostics.
"""
from __future__ import annotations
import csv
import json
import copy
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(r"D:\captures_stop_and_go\forward")
ICP = ROOT / "ICP_Point_to_Plane_after_plane_correction"
OUT = ROOT / "Pin_Based_ICP_Refinement"
CAPTURES = (1, 2, 3, 4)

# ---- Adjustable geometry thresholds (millimetres, degrees) ----
HEIGHT_MIN_MM = 6.0       # points protruding from the board
GRID_MM = 3.0             # segmentation grid; NOT output voxel downsampling
CLOSE_KERNEL = 3
MIN_COMPONENT_AREA_MM2 = 300.0
MIN_COMPONENT_POINTS = 100
MIN_ELONGATION = 1.7     # length / width in board projection
MIN_SLANT_DEG = 5.0     # angle to dominant board axis; rejects near-parallel strips
MAX_SLANT_DEG = 85.0
MAX_PINS_PER_CAPTURE = 2
SHOW_ALL_CANDIDATES = False  # show all height-filtered connected regions
FLIP_BOARD_NORMAL = False  # set True only if protruding pins have negative signed height
SHOW_ORIGINAL_RGB_AFTER_REFINEMENT = True
FACE_RANSAC_MM = 2.5
MIN_FACE_INLIER_RATIO = 0.45
MAX_CANDIDATE_POINTS = 120000
MAX_PAIR_CENTER_MM = 65.0
MAX_PAIR_AXIS_DEG = 22.0
MAX_TRANSLATION_CORRECTION_MM = 35.0
MAX_ROTATION_CORRECTION_DEG = 6.0
MAX_RMS_AFTER_MM = 12.0
MAX_RMS_IMPROVEMENT_FACTOR = 1.05  # require post error < pre error / factor
SHOW_VISUALIZATION = True
SHOW_DETECTION_PER_CAPTURE = True
SHOW_FITTED_RECTANGLES = True
VISUAL_POINT_RADIUS_MM = 3.0
SEED = 24


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), 1e-12)


def frame_from_board(pcd):
    """RANSAC board plane and stable board-plane XY basis."""
    model, ids = pcd.segment_plane(distance_threshold=3.0, ransac_n=3, num_iterations=1500)
    n = unit(model[:3]); xyz = np.asarray(pcd.points)
    center = xyz[np.asarray(ids)].mean(axis=0)
    # PCA on board inliers: the long board direction becomes u.
    subset = xyz[np.asarray(ids)[::max(1, len(ids)//50000)]] - center
    _, _, vt = np.linalg.svd(subset, full_matrices=False)
    u = unit(vt[0] - n*np.dot(vt[0], n))
    v = unit(np.cross(n, u))
    # sign orientation fixed using global X axis (if possible)
    if abs(np.dot(u, [1, 0, 0])) > 0.2 and np.dot(u, [1, 0, 0]) < 0:
        u = -u; v = -v
    # Board plane normal should point toward protruding objects: choose minority side
    heights = (xyz-center) @ n
    if np.percentile(heights, 99) < -np.percentile(heights, 1):
        n = -n; v = -v
    if FLIP_BOARD_NORMAL:
        n = -n
        v = -v  # keep (u, v, n) right-handed
    return center, u, v, n


def uv_of(xyz, origin, u, v):
    d = xyz - origin
    return np.column_stack((d @ u, d @ v))


def components_from_height(xyz, origin, u, v, n):
    """Only height threshold is used to exclude points.

    Connected components group elevated points; no area, elongation or angle
    filtering is applied. Components with fewer than 3 points cannot define
    a plane and are retained for diagnostics but not rectangle fitting.
    """
    height = (xyz-origin) @ n
    keep = np.where(height >= HEIGHT_MIN_MM)[0]
    if len(keep) == 0:
        return [], keep, {"regions": []}
    uv = uv_of(xyz[keep], origin, u, v)
    lo = uv.min(axis=0) - 2 * GRID_MM
    ij = np.floor((uv-lo)/GRID_MM).astype(np.int32)
    dims = ij.max(axis=0) + 3
    if int(dims[0]) * int(dims[1]) > 12_000_000:
        raise RuntimeError("Projection grid too large; increase GRID_MM")
    mask = np.zeros((int(dims[1]), int(dims[0])), np.uint8)
    mask[ij[:, 1], ij[:, 0]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (CLOSE_KERNEL, CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    point_labels = labels[ij[:, 1], ij[:, 0]]
    candidates, regions = [], []
    for lab in range(1, count):
        inds = keep[point_labels == lab]
        if len(inds) == 0:
            continue
        area = float(stats[lab, cv2.CC_STAT_AREA] * GRID_MM**2)
        regions.append({"indices": inds, "status": "accepted", "area": area})
        # Three points are the mathematical minimum for plane fitting.
        if len(inds) < 3:
            continue
        xy = uv[point_labels == lab]
        xy0 = xy - np.median(xy, axis=0)
        cov = np.cov(xy0.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = unit(vecs[:, np.argmax(vals)])
        perpendicular = np.array([-axis[1], axis[0]])
        along, across = xy @ axis, xy @ perpendicular
        length = float(np.percentile(along, 97)-np.percentile(along, 3))
        width = float(np.percentile(across, 97)-np.percentile(across, 3))
        angle = float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0]))))
        candidates.append({
            "indices": inds, "center_uv": np.median(xy, axis=0),
            "axis_uv": axis, "length": length, "width": width,
            "angle": angle, "area": area, "count": len(inds)
        })
    print(f"  Above {HEIGHT_MIN_MM:g} mm: {len(keep)} points; "
          f"connected regions: {len(regions)}; "
          f"regions with >=3 points: {len(candidates)}")
    return candidates, keep, {"regions": regions}

def fit_visible_rectangle(xyz, cand, origin, u, v):
    """Fit actual visible 3D face, then its unconstrained 2D rectangle (no dimensions)."""
    pts = xyz[cand['indices']]
    if len(pts)>MAX_CANDIDATE_POINTS:
        pts=pts[np.linspace(0,len(pts)-1,MAX_CANDIDATE_POINTS).astype(int)]
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = cloud.segment_plane(distance_threshold=FACE_RANSAC_MM,
                                          ransac_n=3,num_iterations=1000)
    ratio=len(inliers)/len(pts)
    if len(inliers) < 3:
        return None
    face=pts[inliers]
    normal=unit(model[:3]); c=face.mean(axis=0)
    # In-plane axes are chosen from the board's longitudinal direction.
    a_raw = u-normal*np.dot(u,normal)
    if np.linalg.norm(a_raw)<1e-6:
        a_raw = v-normal*np.dot(v,normal)
    if np.linalg.norm(a_raw)<1e-6:
        return None
    a=unit(a_raw)
    b=unit(np.cross(normal,a))
    xy=uv_of(face,c,a,b).astype(np.float32)
    rect=cv2.minAreaRect(xy)
    corners2=cv2.boxPoints(rect).astype(float)
    corners3=c + corners2[:,0,None]*a + corners2[:,1,None]*b
    # choose the longer rectangle edge as a direction, sign irrelevant
    sides=np.roll(corners3,-1,axis=0)-corners3
    lengths=np.linalg.norm(sides,axis=1)
    k=int(np.argmax(lengths))
    axis3=unit(sides[k]); center3=corners3.mean(axis=0)
    # 3D axis projected to board frame (for 3DoF registration)
    axis2=unit(np.array([axis3@u,axis3@v]))
    ctr2=uv_of(center3[None,:],origin,u,v)[0]
    plane_rmse=float(np.sqrt(np.mean(((face-c)@normal)**2)))
    # A rectangle may be unreliable when a component includes neighboring geometry.
    return {'center':center3,'center_uv':ctr2,'axis_uv':axis2,
            'corners':corners3,'corners_uv':uv_of(corners3,origin,u,v),'normal':normal,'plane_rmse_mm':plane_rmse,
            'face_inlier_ratio':float(ratio),'length_mm':float(max(lengths)),
            'width_mm':float(min(lengths)), 'point_count':int(len(face)),
            'candidate':cand}


def diagnostic_cloud(pcd, keep, regions, capture):
    """One point cloud, one RGB array: no overlapping geometry / Z-fighting.

    Gray = below height threshold; yellow = height-filtered but not in
    an accepted component; cyan = rejected shape/angle; green/blue = accepted.
    """
    out = copy.deepcopy(pcd)
    xyz = np.asarray(out.points)
    rgb = np.tile(np.array([[0.65, 0.65, 0.65]]), (len(xyz), 1))
    rgb[keep] = [0.98, 0.75, 0.10]
    accepted_palette = [[0.12, 0.9, 0.28], [0.15, 0.4, 1.0], [0.8, 0.2, 0.9]]
    n_accepted = 0
    for r in sorted(regions, key=lambda z: z['area'], reverse=True):
        ids = r['indices']
        if r['status'] == 'shape_or_angle':
            rgb[ids] = [0.15, 0.9, 0.95]
        elif r['status'] == 'accepted':
            rgb[ids] = accepted_palette[n_accepted % len(accepted_palette)]
            n_accepted += 1
    out.colors = o3d.utility.Vector3dVector(rgb)
    print(f'Capture {capture:03d} diagnostic: gray=below 4mm, yellow=height-only/small, '
          f'colored=all elevated connected regions (NO shape filtering). '
          f'Accepted regions: {n_accepted}')
    o3d.visualization.draw_geometries([out],
        window_name=f'Capture {capture:03d} - Diagnostic: height only',
        width=1280, height=800)


def detect(pcd, origin, u, v, n, capture=None):
    xyz = np.asarray(pcd.points)
    candidates, keep, info = components_from_height(xyz, origin, u, v, n)
    if SHOW_ALL_CANDIDATES:
        diagnostic_cloud(pcd, keep, info['regions'], capture if capture is not None else 0)
    fitted = []
    for cand in candidates:
        f = fit_visible_rectangle(xyz, cand, origin, u, v)
        if f is not None:
            fitted.append(f)
        else:
            print(f"  Rejected at face fitting: area={cand['area']:.0f} mm2, "
                  f"points={cand['count']}")
    fitted.sort(key=lambda f: (
        f['candidate']['area'] * min(f['candidate']['length'] / max(f['candidate']['width'], 1), 8)
    ), reverse=True)
    if len(fitted) > MAX_PINS_PER_CAPTURE:
        print(f'  WARNING: {len(fitted)} fitted regions; only top {MAX_PINS_PER_CAPTURE} retained')
    return fitted[:MAX_PINS_PER_CAPTURE], len(candidates)


def axis_difference(a,b):
    return np.degrees(np.arccos(np.clip(abs(float(a@b)),-1,1)))


def match_features(target,source):
    if not target or not source: return []
    cost=np.full((len(target),len(source)),1e8)
    for i,t in enumerate(target):
        for j,s in enumerate(source):
            d=np.linalg.norm(t['center_uv']-s['center_uv'])
            angle=axis_difference(t['axis_uv'],s['axis_uv'])
            if d<=MAX_PAIR_CENTER_MM and angle<=MAX_PAIR_AXIS_DEG:
                cost[i,j]=d+0.6*angle
    ii,jj=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(ii,jj) if cost[i,j]<1e7]


def transform_uv(uv,theta,shift):
    cs,sn=np.cos(theta),np.sin(theta)
    rot=np.array([[cs,-sn],[sn,cs]])
    return uv@rot.T+shift


def refine_pair(target,source,matches):
    """Fit small in-board-plane SE(2) correction with rectangle centers + axes.

    A single long-axis line alone cannot determine motion along its own axis;
    full visible rectangle center is therefore required and checked.
    """
    if not matches: return None,'No reliable corresponding pins'
    def residual(x):
        theta=x[0]; shift=x[1:3]
        r=[]
        for i,j in matches:
            t,s=target[i],source[j]
            transformed=transform_uv(s['center_uv'][None,:],theta,shift)[0]
            # Center contributes translation. Rotation from pin's long-axis orientation.
            r.extend((transformed-t['center_uv']).tolist())
            sa=transform_uv(s['axis_uv'][None,:],theta,[0,0])[0]
            ta=t['axis_uv']
            if sa@ta<0: ta=-ta
            # axis difference expressed as mm-equivalent, with modest weight
            r.extend((12.0*(sa-ta)).tolist())
            # Rectangle geometry: four vertices + four edges. Vertex ordering is
            # resolved by the best cyclic/reversed correspondence at ICP pose.
            tc=t['corners_uv']; sc=s['corners_uv']
            possibilities=[np.roll(sc,k,axis=0) for k in range(4)]
            possibilities += [np.roll(sc[::-1],k,axis=0) for k in range(4)]
            ordered=min(possibilities,key=lambda z:np.sum((z-tc)**2))
            moved=transform_uv(ordered,theta,shift)
            # Downweight inferred corners because missing edges bias minAreaRect.
            r.extend((0.20*(moved-tc)).ravel().tolist())
            # Midpoints of corresponding rectangle edges, also low confidence.
            mid_s=(ordered+np.roll(ordered,-1,axis=0))*0.5
            mid_t=(tc+np.roll(tc,-1,axis=0))*0.5
            r.extend((0.10*(transform_uv(mid_s,theta,shift)-mid_t)).ravel().tolist())
        # regularization to preserve prior ICP, expressed in mm-equivalent
        r.extend([x[0]*70, x[1]*0.08,x[2]*0.08])
        return np.asarray(r)
    before=np.sqrt(np.mean(residual(np.zeros(3))[:-3]**2))
    res=least_squares(residual,np.zeros(3),loss='soft_l1',f_scale=3.0,max_nfev=300)
    x=res.x
    after=np.sqrt(np.mean(residual(x)[:-3]**2))
    if not res.success: return None,'Optimizer failed'
    if abs(np.degrees(x[0]))>MAX_ROTATION_CORRECTION_DEG or np.linalg.norm(x[1:])>MAX_TRANSLATION_CORRECTION_MM:
        return None,'Correction exceeds safety limits'
    if after>MAX_RMS_AFTER_MM or after*MAX_RMS_IMPROVEMENT_FACTOR>=before:
        return None,'No convincing improvement in rectangle agreement'
    return (x,before,after),'OK'


def correction_4x4(theta,shift,origin,u,v,n):
    """Rotate about the board normal through origin, then shift within board plane."""
    cs,sn=np.cos(theta),np.sin(theta)
    R=np.eye(3)*cs+(1-cs)*np.outer(n,n)+sn*np.array([
        [0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    # Important: orientation of (u,v,n) is right-handed.
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=origin-R@origin+shift[0]*u+shift[1]*v
    return T


def serializable(f):
    return {'center_xyz_mm':f['center'].tolist(),'corners_xyz_mm':f['corners'].tolist(),
            'normal':f['normal'].tolist(),'axis_uv':f['axis_uv'].tolist(),
            'length_mm':f['length_mm'],'width_mm':f['width_mm'],
            'plane_rmse_mm':f['plane_rmse_mm'],
            'face_inlier_ratio':f['face_inlier_ratio'],
            'point_count':f['point_count']}


def show_detected_pins(pcd, fitted, capture):
    """Color a SINGLE point cloud; show fitted edges and vertices separately."""
    base = copy.deepcopy(pcd)
    rgb = np.tile(np.array([[0.65, 0.65, 0.65]]), (len(base.points), 1))
    palette = [[0.15, 0.85, 0.25], [0.15, 0.45, 1.0]]
    objects = [base]
    for k, f in enumerate(fitted):
        ids = f['candidate']['indices']
        rgb[ids] = palette[k % len(palette)]
        if SHOW_FITTED_RECTANGLES:
            corners = np.asarray(f['corners'], dtype=float)
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners),
                lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
            )
            line_set.colors = o3d.utility.Vector3dVector([[1, 0.05, 0.05]] * 4)
            objects.append(line_set)
            for point in corners:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=VISUAL_POINT_RADIUS_MM)
                sphere.translate(point)
                sphere.paint_uniform_color([1.0, 0.55, 0.0])
                objects.append(sphere)
    base.colors = o3d.utility.Vector3dVector(rgb)
    print(f'Capture {capture:03d}: showing {len(fitted)} automatically detected pin(s).')
    o3d.visualization.draw_geometries(objects,
        window_name=f'Capture {capture:03d} - Automatic Pin Detection',
        width=1280, height=800)


def main():
    np.random.seed(SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    clouds=[]; original_total=[]
    for capture in CAPTURES:
        p=ICP/f'capture_{capture:03d}_icp_aligned.ply'
        m=ICP/'transformation_matrices'/f'capture_{capture:03d}_total.npy'
        if not p.exists() or not m.exists():
            raise FileNotFoundError(f'Missing ICP output: {p} or {m}')
        clouds.append(o3d.io.read_point_cloud(str(p)))
        original_total.append(np.load(m))
        if clouds[-1].is_empty() or original_total[-1].shape!=(4,4):
            raise ValueError(f'Invalid ICP cloud/matrix for capture {capture}')
    # Use a SINGLE reference frame to avoid independently changing UV axes.
    origin,u,v,n=frame_from_board(clouds[0])
    features=[]; rows=[]
    for i,(cap,cloud) in enumerate(zip(CAPTURES,clouds)):
        f,n_candidates=detect(cloud,origin,u,v,n,cap)
        features.append(f)
        print(f'Capture {cap:03d}: {n_candidates} elevated components, {len(f)} selected faces')
        if SHOW_DETECTION_PER_CAPTURE:
            show_detected_pins(cloud, f, cap)
        (OUT/f'capture_{cap:03d}_detected_pins.json').write_text(
            json.dumps([serializable(x) for x in f],indent=2),encoding='utf-8')
    # Adjacent pair corrections estimated in the SAME ICP global frame.
    # Then compose sequentially; no manual pin naming or clicking.
    accumulated=[np.eye(4)]
    for i in range(1,len(CAPTURES)):
        # transform previous capture features by previously accepted accumulated correction
        Tprev=accumulated[i-1]
        previous=[]
        for f in features[i-1]:
            g=dict(f)
            g['center']=Tprev[:3,:3]@f['center']+Tprev[:3,3]
            g['center_uv']=uv_of(g['center'][None,:],origin,u,v)[0]
            g['corners']=f['corners']@Tprev[:3,:3].T+Tprev[:3,3]
            g['corners_uv']=uv_of(g['corners'],origin,u,v)
            axis3=f['axis_uv'][0]*u+f['axis_uv'][1]*v
            axis3=Tprev[:3,:3]@axis3
            g['axis_uv']=unit(np.array([axis3@u,axis3@v]))
            previous.append(g)
        # Source is in ICP frame. To compare with previous corrected target,
        # use previous accumulated correction as initial transform.
        source=[]
        for f in features[i]:
            g=dict(f)
            g['center']=Tprev[:3,:3]@f['center']+Tprev[:3,3]
            g['center_uv']=uv_of(g['center'][None,:],origin,u,v)[0]
            g['corners']=f['corners']@Tprev[:3,:3].T+Tprev[:3,3]
            g['corners_uv']=uv_of(g['corners'],origin,u,v)
            axis3=Tprev[:3,:3]@(f['axis_uv'][0]*u+f['axis_uv'][1]*v)
            g['axis_uv']=unit(np.array([axis3@u,axis3@v]))
            source.append(g)
        pairs=match_features(previous,source)
        result,reason=refine_pair(previous,source,pairs)
        if result is None:
            accumulated.append(Tprev.copy())
            status=f'SKIPPED: {reason}'
            print(f'{CAPTURES[i-1]:03d}->{CAPTURES[i]:03d}: {status}; matches={len(pairs)}')
            rows.append({'capture':CAPTURES[i],'status':status,'matches':len(pairs),
                         'rms_before_mm':'','rms_after_mm':'','rotation_deg':0,'translation_mm':0})
        else:
            (x,before,after)=result
            step=correction_4x4(x[0],x[1:3],origin,u,v,n)
            accumulated.append(step@Tprev)
            print(f'{CAPTURES[i-1]:03d}->{CAPTURES[i]:03d}: accepted {len(pairs)} pins, RMS {before:.3f}->{after:.3f} mm')
            rows.append({'capture':CAPTURES[i],'status':'ACCEPTED','matches':len(pairs),
                         'rms_before_mm':before,'rms_after_mm':after,
                         'rotation_deg':float(np.degrees(x[0])),
                         'translation_mm':float(np.linalg.norm(x[1:]))})
    for i,(cap,cloud,Tcorr,Told) in enumerate(zip(CAPTURES,clouds,accumulated,original_total)):
        refined=copy.deepcopy(cloud)
        refined.transform(Tcorr)
        o3d.io.write_point_cloud(str(OUT/f'capture_{cap:03d}_pin_refined.ply'),refined)
        np.save(OUT/f'capture_{cap:03d}_pin_correction.npy',Tcorr)
        np.save(OUT/f'capture_{cap:03d}_total.npy',Tcorr@Told)
    with (OUT/'registration_report.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['capture','status','matches','rms_before_mm','rms_after_mm','rotation_deg','translation_mm'])
        writer.writeheader();writer.writerows(rows)
    if SHOW_VISUALIZATION:
        colors=[[1,0.2,0.2],[0.2,0.6,1],[0.2,0.8,0.4],[0.8,0.5,0.9]]
        display=[]
        for cap,color in zip(CAPTURES,colors):
            c=o3d.io.read_point_cloud(str(OUT/f'capture_{cap:03d}_pin_refined.ply'))
            c.paint_uniform_color(color);display.append(c)
        o3d.visualization.draw_geometries(display,window_name='Pin-refined ICP clouds - capture colors')
    if SHOW_ORIGINAL_RGB_AFTER_REFINEMENT:
        rgb_display=[]
        for cap in CAPTURES:
            c=o3d.io.read_point_cloud(str(OUT/f"capture_{cap:03d}_pin_refined.ply"))
            if not c.has_colors():
                print(f"WARNING: capture {cap:03d} has no RGB colors in PLY")
            rgb_display.append(c)
        o3d.visualization.draw_geometries(rgb_display,window_name="Pin-refined clouds - ORIGINAL RGB",width=1280,height=800)
    print('Saved:',OUT)
    print('NOTE: Check detected_pins.json and registration_report.csv before trusting results.')

if __name__=='__main__':
    main()
