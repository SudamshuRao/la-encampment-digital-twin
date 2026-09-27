# Enriched Hexagon Variable Dictionary

This dictionary describes the columns in the enriched CSV files:

- `half_variables_and_tents_geo_enriched.csv`
- `quarter_variables_and_tents_geo_enriched.csv`
- `three_fourth_variables_and_tent_geo_enriched.csv`

The three CSVs each contain 100 columns. Across all three files there are 101 unique column names because `ACSHHPBLPV_P` appears in the three-fourth layer while `ACSBLWPV_P` appears in the half and quarter layers.

## Naming Notes

- `_count`: count of features inside the hexagon.
- `_length_m`: total line length inside the hexagon, in meters.
- `nearest_*_m`: distance from the hexagon centroid to the nearest feature, in meters.
- `_area_pct`: percent of the hexagon area overlapped by a polygon layer.
- `_intersects`: binary flag, `1` if the hexagon intersects the layer, else `0`.
- `_CY`: Esri current-year estimate.
- `_FY`: Esri forecast-year estimate.
- `_P`: percent.
- `_I`: index, usually relative to a national average of 100 in Esri data.

## Base Hexagon / Join Fields

| Variable | Meaning |
|---|---|
| `OBJECTID` | Unique ID for each hexagon. Primary key used for joins. |
| `Join_Count` | Count produced by the previous GIS spatial join. In this dataset it reflects the number of joined tent/detection records from the prior layer build. |
| `TARGET_FID` | GIS join artifact: feature ID of the target hexagon in the previous spatial join. |
| `JOIN_FID` | GIS join artifact: feature ID of the joined source feature in the previous spatial join. |
| `GRID_ID` | Hexagon grid label or identifier. Useful for map referencing. |
| `hex_area_sqkm` | Area of the hexagon in square kilometers, calculated after projection to a meter-based CRS. |

## Esri Enrichment Metadata

| Variable | Meaning |
|---|---|
| `ID` | Esri GeoEnrichment area ID from the first enrichment pass. |
| `sourceCountry` | Country code/source country for the first Esri enrichment pass. |
| `ENRICH_FID` | Esri enrichment feature ID from the first enrichment pass. |
| `aggregationMethod` | Esri method used to apportion/enrich data into the polygon. |
| `populationToPolygonSizeRating` | Esri quality/confidence indicator comparing population allocation to polygon size. |
| `apportionmentConfidence` | Esri confidence score for how reliably variables were apportioned to the hexagon. |
| `HasData` | Esri flag indicating whether enrichment data was available for the hexagon. |
| `ID_1` | Esri GeoEnrichment area ID from the second enrichment pass. |
| `sourceCountry_1` | Country code/source country for the second Esri enrichment pass. |
| `ENRICH_FID_1` | Esri enrichment feature ID from the second enrichment pass. |
| `aggregationMethod_1` | Esri method used in the second enrichment pass. |
| `populationToPolygonSizeRating_1` | Esri quality/confidence indicator from the second enrichment pass. |
| `apportionmentConfidence_1` | Esri apportionment confidence score from the second enrichment pass. |
| `HasData_1` | Esri flag indicating whether second-pass enrichment data was available. |

## Socioeconomic / Esri / ACS Fields

| Variable | Meaning |
|---|---|
| `CRMCYBURG` | Esri current-year burglary crime estimate/count. |
| `CRMCYTOTC` | Esri current-year total crime estimate/count. |
| `TOTPOP_CY` | Esri current-year total population estimate. |
| `UNEMPRT_CY_I` | Esri current-year unemployment-rate index. Values above 100 indicate higher unemployment than the national baseline. |
| `HISPPOP_CY_P` | Esri current-year percent Hispanic/Latino population. |
| `MEDHINC_CY_I` | Esri current-year median household income index. Values above 100 indicate higher median income than the national baseline. |
| `ACSVET_P` | ACS percent of population that are veterans. |
| `ACSVET_P_1` | Duplicate/second-pass ACS veteran percentage field. |
| `ACSHSGRAD_P` | ACS percent of population with high school graduate as educational attainment. |
| `ACS35NOHI_P` | ACS percent of households or population with no health insurance. |
| `ACSEDNOSCH_P` | ACS percent of population with no schooling completed. |
| `ACSGRNTI50_P` | ACS percent of renter households with gross rent at least 50 percent of household income. Severe rent burden proxy. |
| `ACSBLWPV_P` | ACS percent of population below poverty level. Present in half and quarter enriched layers. |
| `ACSHHPBLPV_P` | ACS percent of households below poverty level. Present in the three-fourth enriched layer. |
| `ACSPUBAI_P` | ACS percent of population with public assistance income. |
| `ACSHHDIS_P` | ACS percent of households with disability-related indicator. |
| `X14068_X_I` | Esri market-potential index field for item code `14068`, current period. Exact item label was not available in the local metadata. Treat as an Esri index field where 100 is the national average. |
| `X14068FY_X_I` | Esri market-potential index field for item code `14068`, forecast year. Exact item label was not available in the local metadata. Treat as an Esri index field where 100 is the national average. |

## Original Tent Detection Fields

| Variable | Meaning |
|---|---|
| `source_run` | Pipeline/run folder or batch that produced the original tent detection record. |
| `latitude` | Latitude of the joined tent detection/reference point. |
| `longitude` | Longitude of the joined tent detection/reference point. |
| `address` | Address associated with the detection point, when available. |
| `heading` | Street-view image heading in degrees. |
| `filename` | Image filename associated with the tent detection. |
| `tent_prediction_count` | Number of tent predictions detected for the joined image/location. |
| `max_conf` | Maximum model confidence score for tent detections at that image/location. |
| `classes_seen` | Object classes detected by the vision model, usually `tent`. |

## Reference Tent Aggregates

These were recomputed from `tent_detections_full_18060_combined.csv` during the enrichment run.

| Variable | Meaning |
|---|---|
| `tent_ref_point_count` | Number of reference detection points falling inside the hexagon. |
| `tent_ref_prediction_sum` | Sum of `tent_prediction_count` for all reference detection points inside the hexagon. |
| `tent_ref_max_conf` | Highest tent detection confidence among reference points inside the hexagon. |
| `tent_ref_mean_conf` | Mean tent detection confidence among reference points inside the hexagon. |

## Climate / Land Cover / Heat Fields

| Variable | Meaning |
|---|---|
| `climate_t2m_2024_c` | NASA POWER 2024 mean monthly 2-meter air temperature at the union-grid centroid, in Celsius. Same value applied to all hexes. |
| `climate_tmax_2024_c` | NASA POWER 2024 mean monthly maximum 2-meter air temperature at the union-grid centroid, in Celsius. Same value applied to all hexes. |
| `climate_ppt_2024_mm` | NASA POWER 2024 total precipitation at the union-grid centroid, in millimeters. Same value applied to all hexes. |
| `tree_canopy_proxy_pct` | Percent of sampled LARIAC landcover pixels classified as tree canopy. Proxy for tree canopy coverage. |
| `vegetation_proxy_pct` | Percent of sampled LARIAC landcover pixels classified as vegetation or canopy. |
| `impervious_proxy_pct` | Percent of sampled LARIAC landcover pixels classified as impervious surface. |
| `water_proxy_pct` | Percent of sampled LARIAC landcover pixels classified as water. |
| `bare_ground_proxy_pct` | Percent of sampled LARIAC landcover pixels classified as bare ground. |
| `dominant_landcover_class` | Most common LARIAC landcover class in the hexagon. Classes: `1` canopy non-deciduous, `2` medium vegetation, `3` low vegetation, `4` bare ground, `5` impervious, `6` water, `7` canopy deciduous. |
| `heat_exposure_proxy_score` | Heat exposure proxy calculated from high imperviousness and low canopy. Higher values indicate hotter, less shaded built environments. |
| `heat_exposure_area_pct` | Percent of hexagon overlapped by SCAG heat exposure polygons. The SCAG endpoint failed during the run, so this is currently empty/zero. |
| `heat_exposure_intersects` | Binary flag for SCAG heat exposure polygon intersection. The SCAG endpoint failed during the run, so this is currently empty/zero. |

## Facility / Service Counts

| Variable | Meaning |
|---|---|
| `transit_stops_count` | Count of LA Metro transit stops inside the hexagon. |
| `hospital_count` | Count of hospitals or medical centers inside the hexagon. |
| `fire_station_count` | Count of fire stations inside the hexagon. |
| `library_count` | Count of libraries inside the hexagon. |
| `college_count` | Count of colleges/universities inside the hexagon. |
| `public_school_count` | Count of public schools inside the hexagon. |
| `private_school_count` | Count of private schools inside the hexagon. |
| `recreation_center_count` | Count of recreation centers inside the hexagon. |
| `adult_recreation_count` | Count of adult recreation program locations inside the hexagon. |
| `cool_zone_count` | Count of cooling centers/cool zones inside the hexagon. |
| `affordable_housing_count` | Count of public housing locations used as an affordable-housing proxy. |
| `sud_treatment_count` | Count of substance-use-disorder treatment facilities inside the hexagon. |
| `business_sites_count` | Count of active business sites inside the hexagon. |
| `gas_station_count` | Count of gas stations/fuel-service businesses inside the hexagon. |
| `police_station_count` | Count of LAPD police stations inside the hexagon. |
| `food_access_count` | Count of food access sites inside the hexagon. |
| `lausd_school_site_count` | Count of LAUSD school sites inside the hexagon. |
| `osm_amenity_count` | Count of OpenStreetMap amenities inside the hexagon, including shelters, social facilities, toilets, drinking water, clinics, hospitals, pharmacies, restaurants, and fast food. |

## Road / Freeway / Waterway Fields

| Variable | Meaning |
|---|---|
| `osm_major_road_segments_count` | Number of OSM major-road line segments intersecting the hexagon. Includes motorway, trunk, primary, and related links. |
| `osm_major_road_length_m` | Total OSM major-road length inside the hexagon, in meters. |
| `nearest_osm_major_road_m` | Distance from hexagon centroid to nearest OSM major road, in meters. |
| `freeway_segments_count` | Number of OSM freeway/trunk line segments intersecting the hexagon. |
| `freeway_length_m` | Total OSM freeway/trunk length inside the hexagon, in meters. |
| `nearest_freeway_m` | Distance from hexagon centroid to nearest OSM freeway/trunk, in meters. |
| `caltrans_highway_segments_count` | Number of Caltrans State Highway Network segments intersecting the hexagon. |
| `caltrans_highway_length_m` | Total Caltrans highway length inside the hexagon, in meters. |
| `nearest_caltrans_highway_m` | Distance from hexagon centroid to nearest Caltrans highway segment, in meters. |
| `river_drainage_segments_count` | Number of OSM waterway/drainage line segments intersecting the hexagon. |
| `river_drainage_length_m` | Total OSM waterway/drainage length inside the hexagon, in meters. |
| `nearest_river_drainage_m` | Distance from hexagon centroid to nearest OSM river, stream, canal, drain, or related waterway, in meters. |

## Flood / Hazard Overlay Fields

| Variable | Meaning |
|---|---|
| `fema_flood_hazard_area_pct` | Percent of the hexagon overlapped by FEMA NFHL flood hazard polygons. FEMA returned partial data before a server error; see source summary. |
| `fema_flood_hazard_intersects` | Binary flag, `1` if the hexagon intersects FEMA NFHL flood hazard polygons. FEMA returned partial data before a server error; see source summary. |
| `la_geohub_100yr_flood_area_pct` | Percent of hexagon overlapped by LA County/GeoHub 100-year flood plain polygons. |
| `la_geohub_100yr_flood_intersects` | Binary flag for intersection with LA County/GeoHub 100-year flood plain. |
| `la_geohub_500yr_flood_area_pct` | Percent of hexagon overlapped by LA County/GeoHub 500-year flood plain polygons. |
| `la_geohub_500yr_flood_intersects` | Binary flag for intersection with LA County/GeoHub 500-year flood plain. |
| `la_geohub_fire_hazard_area_pct` | Percent of hexagon overlapped by LA County fire hazard severity zones. |
| `la_geohub_fire_hazard_intersects` | Binary flag for intersection with LA County fire hazard severity zones. |

## Source Summary

Detailed source status for downloads and joins is stored in:

`geojson/geographic_cache/geo_feature_source_summary.json`

Important limitations from the enrichment run:

- SCAG heat exposure service returned server error `500`, so direct SCAG heat columns are empty/zero.
- FEMA NFHL returned partial data before a server error. FEMA columns are useful but should be treated as partial public-source coverage.
- `X14068_X_I` and `X14068FY_X_I` are Esri market-potential index fields, but the exact item label for code `14068` was not available from local metadata.
