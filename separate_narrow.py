import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio.features

from rasterio.transform import Affine
from skimage.morphology import binary_opening, disk
from shapely.geometry import shape, LineString, Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union


def create_gdf(poly, narrow):
    gdf = gpd.GeoDataFrame(geometry=[poly])
    gdf['narrow_part'] = [narrow for _ in range(len(gdf))]
    return gdf


def opening_poly(poly, resolution=0.5, radius=10, buffer=3):
    # rasterisation of input polygon
    bounds = poly.bounds
    transform = Affine(resolution, 0, bounds[0], 0, -resolution, bounds[3])
    x_range = int((bounds[2] - bounds[0]) / resolution)
    y_range = int((bounds[3] - bounds[1]) / resolution)

    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    out_shape = (y_range, x_range)
    poly_rasterized = rasterio.features.rasterize([poly], out_shape=out_shape, transform=transform)

    # calculate opening
    radius_pixels = int(radius / resolution)
    footprint = disk(radius_pixels)
    poly_opening = binary_opening(poly_rasterized, footprint)

    # vectorize output opening
    shapes = rasterio.features.shapes(poly_opening.astype('int16'), transform=transform)
    shapes = [shape(g) for g, v in shapes if v == 1]
    opening_gdf = gpd.GeoDataFrame(geometry=shapes)
    opening_gdf = opening_gdf.dissolve()
    if len(opening_gdf) == 1:
        opening_gdf = opening_gdf.simplify(tolerance=1)
        opening_gdf = opening_gdf.buffer(buffer)
        opening_geom = opening_gdf.geometry.item()
    else:
        opening_geom = None

    return opening_geom


def check_neighbouring_lines(lines):
    for line in lines:
        lines_dist = lines.copy()
        lines_dist.remove(line)
        distances = line.distance(lines_dist)
        if distances.min() == 0:
            return True
    return False


def merge_touching_lines(line_left, line_right):
    coords_left = [coords for coords in line_left.coords]
    coords_right = [coords for coords in line_right.coords]

    coords_left_first = coords_left[0]
    coords_left_last = coords_left[-1]
    coords_right_first = coords_right[0]
    coords_right_last = coords_right[-1]

    coords = []
    if coords_left_first == coords_right_first:
        for xy in coords_left[::1]:
            coords.append(xy)
        for xy in coords_right[1:]:
            coords.append(xy)

    elif coords_left_first == coords_right_last:
        for xy in coords_right:
            coords.append(xy)
        for xy in coords_left[1:]:
            coords.append(xy)

    elif coords_left_last == coords_right_first:
        for xy in coords_left:
            coords.append(xy)
        for xy in coords_right[1:]:
            coords.append(xy)

    else:
        for xy in coords_left[:-1]:
            coords.append(xy)
        for xy in coords_right[::1]:
            coords.append(xy)

    line = LineString(coords)
    return line


def polygons_from_multilinestring(multilinestring):
    lines = [line for line in multilinestring.geoms]

    neighbouring_lines = check_neighbouring_lines(lines)
    while neighbouring_lines:
        for line in lines:
            lines_run = lines.copy()
            lines_run.remove(line)

            distances = line.distance(lines_run)
            indices = np.nonzero(distances == 0)[0]
            neighbours = np.array(lines_run)[indices]

            if len(neighbours) > 0:
                lines.remove(line)
                for neighbour in neighbours:
                    line = merge_touching_lines(line, neighbour)
                    lines.remove(neighbour)
                lines.append(line)
                break

        if len(lines) == 1:
            neighbouring_lines = False
        else:
            neighbouring_lines = check_neighbouring_lines(lines)

    polygons = [Polygon(line) for line in lines if len([c for c in line.coords]) > 2]

    return polygons


def polygon_from_linestring(line):
    coords = [c for c in line.coords]
    if len(coords) > 2:
        poly = Polygon(coords)
        return poly


def merge_small_polygons(gdf_left, gdf_right, min_area):
    # filter small polygons and merge them with neighbouring ones
    gdf_left['area'] = gdf_left.area
    gdf_right['area'] = gdf_right.area

    gdf_left['area'] = gdf_left.area
    gdf_right['area'] = gdf_right.area
    gdf_add_left = gdf_right[gdf_right['area'] < min_area]
    gdf_keep_left = gdf_left[gdf_left['area'] >= min_area]
    gdf_add_right = gdf_left[gdf_left['area'] < min_area]
    gdf_keep_right = gdf_right[gdf_right['area'] >= min_area]

    gdf_right = pd.concat([gdf_keep_right, gdf_add_right])
    gdf_right = gdf_right.drop(columns=['area'])
    gdf_right = gdf_right.dissolve().explode(index_parts=False)

    gdf_left = pd.concat([gdf_keep_left, gdf_add_left])
    gdf_left = gdf_left.drop(columns=['area'])
    gdf_left = gdf_left.dissolve().explode(index_parts=False)

    return gdf_left, gdf_right


def merge_nonlinear_features(gdf_left, gdf_right, max_touching_part):
    gdf_right = gdf_right.dissolve().explode(index_parts=False).reset_index(drop=True)
    gdf_left = gdf_left.dissolve()

    # filter according to percentage of perimeter touching with neighbouring polygon
    geoms_right = gdf_right.geometry
    left_geom = gdf_left.geometry.item()
    touches = [geom.intersection(left_geom) for geom in geoms_right]
    touch_len = [touch.length for touch in touches]
    perimeter = gdf_right.length
    gdf_right['touch_part'] = pd.Series(touch_len) / perimeter

    # filter according to percentage of perimeter touching with neighbouring polygon
    gdf_add_left = gdf_right[gdf_right['touch_part'] > max_touching_part]
    gdf_add_left = gdf_add_left.drop(columns=['touch_part'])
    gdf_right = gdf_right[gdf_right['touch_part'] < max_touching_part]
    gdf_right['narrow_part'] = [True for _ in range(len(gdf_right))]
    gdf_right['narrow_part'] = gdf_right['narrow_part'].astype(bool)

    gdf_left = pd.concat([gdf_left, gdf_add_left])
    gdf_left = gdf_left.dissolve().explode(index_parts=False)
    gdf_left['narrow_part'] = [False for _ in range(len(gdf_left))]
    gdf_left['narrow_part'] = gdf_left['narrow_part'].astype(bool)

    # create union of narrow and broad parts geodataframes
    gdf_all = pd.concat([gdf_right, gdf_left])
    gdf_all = gdf_all.drop(columns=['touch_part'])

    return gdf_all


def straight_touch_lines(gdf_left, gdf_right):
    geom_left = gdf_left.dissolve().geometry.item()
    geom_right = gdf_right.dissolve().geometry.item()

    # make clip lines straight
    touches = [geom.intersection(geom_left) for geom in gdf_right.geometry]
    touches_collection = [[g for g in geom.geoms if g.geom_type != 'Point'] for geom in touches if
                          geom.geom_type == 'GeometryCollection']
    touches_collection = [item for row in touches_collection for item in row]
    touches_other = [geom for geom in touches if geom.geom_type != 'GeometryCollection']
    touches = [item for row in [touches_collection, touches_other] for item in row]
    touches = [touch for touch in touches if touch.length > 0]
    poly_corrections = [
        polygons_from_multilinestring(touch) if touch.geom_type == 'MultiLineString' else polygon_from_linestring(touch)
        for touch in touches]
    poly_corrections = [poly for poly in poly_corrections if poly is not None]
    poly_corrections = [item for row in poly_corrections for item in row]
    poly_corrections = [make_valid(poly) for poly in poly_corrections]
    poly_corrections_poly = [poly for poly in poly_corrections if poly.geom_type == 'Polygon']
    poly_corrections_multipoly = [poly for poly in poly_corrections if poly.geom_type == 'MultiPolygon']
    poly_corrections_multipoly = [[poly for poly in multipoly.geoms] for multipoly in poly_corrections_multipoly]
    poly_corrections_multipoly = [item for row in poly_corrections_multipoly for item in row]
    poly_corrections = [item for row in [poly_corrections_poly, poly_corrections_multipoly] for item in row]
    correction_geom = unary_union(poly_corrections)

    add_right = geom_left.intersection(correction_geom)
    add_left = geom_right.intersection(correction_geom)

    if add_right.geom_type == 'GeometryCollection':
        geometry = [geom for geom in add_right.geoms]
        geometry = [geom for geom in geometry if ((geom.geom_type == 'Polygon') or (geom.geom_type == 'MultiPolygon'))]
        add_right = unary_union(geometry)
    if add_left.geom_type == 'GeometryCollection':
        geometry = [geom for geom in add_left.geoms]
        geometry = [geom for geom in geometry if ((geom.geom_type == 'Polygon') or (geom.geom_type == 'MultiPolygon'))]
        add_left = unary_union(geometry)

    geom_right = geom_right.difference(add_left)
    geom_right = geom_right.union(add_right)
    geom_left = geom_left.union(add_left)
    geom_left = geom_left.difference(add_right)

    gdf_left = gpd.GeoDataFrame(geometry=[geom_left]).dissolve().explode(index_parts=False)
    gdf_right = gpd.GeoDataFrame(geometry=[geom_right]).dissolve().explode(index_parts=False)

    return gdf_left, gdf_right


def split_narrow_geom(geom_left, geom_right, min_area=15, max_touching_part=0.2):
    # calculate intersection and difference between input polygons
    if not geom_left.is_valid:
        geom_left = make_valid(geom_left)
    if not geom_right.is_valid:
        geom_right = make_valid(geom_right)
    diff = geom_right.difference(geom_left)

    if diff.geom_type == 'MultiPolygon':
        diff_geoms = [geom for geom in diff.geoms]
    elif diff.geom_type == 'GeometryCollection':
        diff_list = [geom for geom in diff.geoms]
        diff_geoms_poly = [geom for geom in diff_list if geom.geom_type == 'Polygon']
        diff_geoms_multipoly = [geom for geom in diff_list if geom.geom_type == 'MultiPolygon']
        if len(diff_geoms_multipoly) > 0:
            diff_geoms_multipoly = [[poly for poly in geom.geoms] for geom in diff_geoms_multipoly]
            diff_geoms_multipoly = [item for row in diff_geoms_multipoly for item in row]
        diff_geoms = [item for row in [diff_geoms_poly, diff_geoms_multipoly] for item in row]

    else:
        diff_geoms = [diff]

    gdf_diff = gpd.GeoDataFrame(geometry=diff_geoms)

    intersection = geom_right.intersection(geom_left)
    if intersection.geom_type == 'Polygon':
        gdf_intersection = gpd.GeoDataFrame(geometry=[intersection])
    else:
        intersection_geoms = [geom for geom in intersection.geoms if geom.geom_type == 'Polygon']
        gdf_intersection = gpd.GeoDataFrame(geometry=intersection_geoms).dissolve()

    # geometry_corrections
    gdf_intersection, gdf_diff = merge_small_polygons(gdf_intersection, gdf_diff, min_area)
    if len(gdf_diff) == 0:
        gdf = create_gdf(geom_right, False)
    elif len(gdf_intersection) == 0:
        gdf = create_gdf(geom_right, True)
    else:
        gdf_intersection, gdf_diff = straight_touch_lines(gdf_intersection, gdf_diff)
        if len(gdf_diff) == 0:
            gdf = create_gdf(geom_right, False)
        elif len(gdf_intersection) == 0:
            gdf = create_gdf(geom_right, True)
        else:
            gdf_intersection, gdf_diff = merge_small_polygons(gdf_intersection, gdf_diff, min_area)
            if len(gdf_diff) == 0:
                gdf = create_gdf(geom_right, False)
            elif len(gdf_intersection) == 0:
                gdf = create_gdf(geom_right, True)
            else:
                gdf = merge_nonlinear_features(gdf_intersection, gdf_diff, max_touching_part)
    return gdf


def cut_poly(poly, resolution=0.5, radius=10, buffer=3, min_area=15, max_touching_part=0.2):
    opening_geom = opening_poly(poly, resolution=resolution, radius=radius, buffer=buffer)
    if opening_geom is None:
        gdf = create_gdf(poly, True)
    elif opening_geom.intersection(poly).area == poly.area:
        gdf = create_gdf(poly, False)
    else:
        gdf = split_narrow_geom(opening_geom, poly, min_area=min_area, max_touching_part=max_touching_part)

    return gdf


def narrow_wide_geometries(geoseries, resolution=0.5, radius=10, buffer=3, min_area=15, max_touching_part=0.2):
    gdfs = [cut_poly(poly, resolution=resolution, radius=radius, buffer=buffer, min_area=min_area,
                     max_touching_part=max_touching_part) for poly in geoseries]

    gdf = gpd.GeoDataFrame()
    for gdf_poly in gdfs:
        gdf = pd.concat([gdf, gdf_poly])
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf
