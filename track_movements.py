def track_animal_movements_adjusted(data, shapefile_path):
    """
    Adjusted function to track the movement of animals, considering temporary exits from polygons.
    """
    # Step 1: Load polygons from the provided shapefile
    polygons_gdf = gpd.read_file(shapefile_path)
    polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    # Step 2: Create Point geometries using vectorized function
    data['geometry'] = gpd.points_from_xy(data.longitude, data.latitude)

    # Convert data to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

    # Step 3: Perform a spatial join to get the polygon name and ID for each point
    points_polygons = gpd.sjoin(points_gdf, polygons_gdf, how='left', predicate='within')

    # Step 4: Sort the data by 'animal_id' and 'timestamp'
    points_polygons = points_polygons.sort_values(by=['animal_id', 'timestamp']).reset_index(drop=True)

    # Ensure 'timestamp' is datetime
    if not pd.api.types.is_datetime64_any_dtype(points_polygons['timestamp']):
        points_polygons['timestamp'] = pd.to_datetime(points_polygons['timestamp'])

    # Step 5: For each animal, fill null 'id's only when previous and next 'id's are the same
    def fill_ids(group):
        group = group.copy()
        group['id_ffill'] = group['id'].ffill()
        group['id_bfill'] = group['id'].bfill()
        # Condition: null 'id's where both forward fill and backward fill are the same and not null
        condition = group['id'].isnull() & (group['id_ffill'] == group['id_bfill']) & group['id_ffill'].notnull()
        group['filled_id'] = group['id']
        group.loc[condition, 'filled_id'] = group.loc[condition, 'id_ffill']
        # For 'name' as well
        group['filled_name'] = group['name']
        group.loc[condition, 'filled_name'] = group.loc[condition, 'name'].ffill()
        return group

    points_polygons = points_polygons.groupby('animal_id', group_keys=False).apply(fill_ids)

    # Step 6: Detect polygon changes based on 'filled_id'
    points_polygons['prev_filled_id'] = points_polygons.groupby('animal_id')['filled_id'].shift()
    points_polygons['polygon_change'] = (
    (points_polygons['filled_id'] != points_polygons['prev_filled_id']) &
    (points_polygons['filled_id'].notnull())
    )

    # For the first occurrence, set polygon_change to True if filled_id is not null
    first_indices = points_polygons.groupby('animal_id').head(1).index
    points_polygons.loc[first_indices, 'polygon_change'] = points_polygons.loc[first_indices, 'filled_id'].notnull()

    # Step 7: Assign a 'stay_id' to each continuous stay in a polygon
    points_polygons['stay_id'] = points_polygons.groupby('animal_id')['polygon_change'].cumsum()

    # Step 8: Extract date from timestamp for active days calculation
    points_polygons['date'] = points_polygons['timestamp'].dt.floor('D')

    # Step 9: Group by 'animal_id' and 'stay_id' and compute aggregations
    # Consider only the rows where 'filled_id' is not null
    valid_points = points_polygons[points_polygons['filled_id'].notnull()]

    grouped = valid_points.groupby(['animal_id', 'stay_id'])

    result = grouped.agg(
    polygon_name=('filled_name', 'first'),
    polygon_id=('filled_id', 'first'),
    first_seen_date=('timestamp', 'min'),
    last_seen_date=('timestamp', 'max'),
    active_days=('date', 'nunique')
    ).reset_index()