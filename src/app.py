"""
Author: Mohanad Albughdadi
"""

import base64
import logging
import os
import zipfile
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from sam_utils import generate_automatic_mask, sam_prompt_bbox
from utils import (
    bounds_to_table_row,
    id_generator,
    reproject_to_epsg4326,
    shape_to_table_row,
    download_from_wms,
)
from config import WMS_URL, LAYER, IMAGE_FORMAT, WORK_DIR, RESOLUTION, DEBUG

os.makedirs(WORK_DIR, exist_ok=True)  # type: ignore

annotation_types = ["ROI BBox", "Object BBox", "Foreground Point", "Background Point"]

columns = ["id", "type", "x_min", "y_min", "x_max", "y_max"]
models = ["sam_vit_b_01ec64.pth", "sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth"]


external_stylesheets = [dbc.themes.BOOTSTRAP, "src/assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)  # type: ignore
app.title = "Satellite SAM"

server = app.server

# Define the WMS layer configuration
wms_layer = dl.WMSTileLayer(
    url="https://tiles.maps.eox.at/wms",
    layers="s2cloudless-2020_3857",
    format="image/jpeg",
    transparent=True,
    attribution="Sentinel-2 cloudless layer for 2020 by EOX",
)

esri_tile_layer = dl.TileLayer(
    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution="Esri",
)

# Cards
image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H2("Satellite Map")),
        dbc.CardBody(
            [
                dl.Map(
                    children=[
                        dl.LayersControl(
                            [
                                dl.Overlay(dl.TileLayer(), name="OSM", checked=True),
                                dl.Overlay(wms_layer, name="Sentinel-2", checked=True),
                                dl.Overlay(esri_tile_layer, name="ESRI", checked=True),
                            ],
                            id="layers-control",
                            collapsed=True,
                        ),
                        dl.FeatureGroup(
                            [
                                dl.LocateControl(
                                    options={
                                        "locateOptions": {"enableHighAccuracy": True}
                                    }
                                ),
                                dl.MeasureControl(
                                    position="topleft",
                                    primaryLengthUnit="kilometers",
                                    primaryAreaUnit="sqmeters",
                                    id="measure_control",
                                ),
                                dl.EditControl(
                                    id="edit_control",
                                    draw={
                                        "polyline": False,
                                        "polygon": False,
                                        "circle": False,
                                        "circlemarker": False,
                                    },
                                ),
                            ]
                        ),
                    ],
                    id="map",
                    style={
                        "width": "100%",
                        "height": "66vh",
                        "margin": "auto",
                        "display": "block",
                    },
                    center=[25.1190, 55.1318],
                    zoom=12,
                )
            ],
            id="map-card",
        ),
        dbc.CardFooter([]),
    ],
)
upload_data_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Data upload")),
        dbc.CardBody(
            [
                dcc.Upload(
                    id="upload-tif",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=False,  # Set to True if multiple file upload is needed
                ),
                dcc.Store(id="output-data-upload", data=None),
            ]
        ),
    ]
)
annotated_data_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Annotated data")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H3("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation=(
                                            "dropdown" if n == "type" else "input"
                                        ),
                                    )
                                    for n in columns
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={
                                    "overflow": "visible",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown={
                                    "type": {
                                        "options": [
                                            {"label": o, "value": o}
                                            for o in annotation_types
                                        ],
                                        "clearable": False,
                                    }
                                },
                                style_cell_conditional=[
                                    {
                                        "if": {"column_id": "type"},
                                        "textAlign": "left",
                                    }
                                ],
                                fill_width=True,
                                css=[
                                    {
                                        "selector": ".Select-menu-outer",
                                        "rule": "display: block !important",
                                    }
                                ],
                            ),
                        ],
                    ),
                ),
            ]
        ),
    ]
)
model_card = dbc.Card(
    [
        dbc.CardHeader([html.H2("SAM Configuration")]),
        dbc.CardBody(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Stack(
                                [
                                    html.H4("Model Type"),
                                    dcc.Dropdown(
                                        id="sam-model",
                                        options=[
                                            {
                                                "label": "_".join(t.split("_")[0:3]),
                                                "value": t,
                                            }
                                            for t in models
                                        ],
                                        value=models[0],
                                        clearable=False,
                                    ),
                                    html.Hr(),
                                    html.H4("Automatic Mask Configuration"),
                                    html.H5("Predicition IoU Threshold"),
                                    dcc.Input(
                                        id="pred-iou-thresh",
                                        type="number",
                                        placeholder="IoU threshold: input between 0 and 1",
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        value=0.88,
                                    ),
                                    html.H5("Stability Score Threshold"),
                                    dcc.Input(
                                        id="stability-score-thresh",
                                        type="number",
                                        placeholder="Stability score threshold: input between 0 and 1",
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        value=0.95,
                                    ),
                                ],
                                gap=3,
                            )
                        ],
                        align="center",
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            [
                html.Div(
                    [
                        dbc.Button(
                            "Segment ROI",
                            id="segment-button",
                            outline=True,
                            color="primary",
                            n_clicks=0,
                            style={"horizontalAlign": "left"},
                            className="me-md-2",
                        ),
                        html.Div(id="dummy1", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can run the SAM model by clicking on this button",
                            target="segment-button",
                        ),
                        dbc.Button(
                            "Download Results",
                            id="download-button",
                            outline=True,
                            color="info",
                            n_clicks=0,
                            disabled=True,
                            style={"horizontalAlign": "middle"},
                            className="me-md-2",
                        ),
                        html.Div(id="dummy2", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can download the results by clicking here",
                            target="download-button",
                        ),
                        dcc.Download(
                            id="download-link",
                        ),
                        dbc.Button(
                            "Refresh",
                            id="refresh-button",
                            outline=True,
                            color="success",
                            n_clicks=0,
                            style={"horizontalAlign": "right"},
                            className="me-md-2",
                        ),
                        dbc.Tooltip(
                            "Click here to refresh page and segment a new zone",
                            target="refresh-button",
                        ),
                        dcc.Location(id="url", refresh=True),
                    ],
                    className="d-grid gap-2 d-md-flex justify-content-center",
                )
            ]
        ),
    ],
)
# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("Satellite Segment Anything")),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-5",
)

app.layout = html.Div(
    [
        dbc.Spinner(
            id="loading-1",
            type="grow",
            color="success",
            children=[
                dcc.Store(id="downloaded_image_path"),
                dcc.Store(id="prev-table-data"),
                navbar,
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    image_annotation_card,
                                    md=7,
                                ),
                                dbc.Col(
                                    dbc.Stack(
                                        [
                                            upload_data_card,
                                            annotated_data_card,
                                            model_card,
                                        ]
                                    ),
                                    md=5,
                                ),
                            ],
                        ),
                    ],
                    fluid=True,
                ),
            ],
        )
    ]
)


@app.callback(
    Output("url", "href"),
    [Input("refresh-button", "n_clicks"), Input("downloaded_image_path", "data")],
    prevent_initial_call=True,
)
def refresh_page(n_clicks, downloaded_imgs):
    if n_clicks is not None and n_clicks > 0:
        if downloaded_imgs is not None:
            for img in downloaded_imgs:
                os.remove(img)
        return "/"
    return dash.no_update


@app.callback(
    [
        Output("output-data-upload", "data"),
        Output("map-card", "children", allow_duplicate=True),
    ],
    [
        Input("upload-tif", "contents"),
        Input("upload-tif", "filename"),
    ],
    prevent_initial_call="initial_duplicate",
)
def update_output_upload(contents, filename):
    logging.info(filename)
    if contents is None and filename is None:
        raise PreventUpdate
    _, content_string = contents.split(",")
    process_id = id_generator()
    process_path = os.path.join(WORK_DIR, process_id)  # type: ignore
    os.makedirs(process_path)
    decoded = base64.b64decode(content_string)
    file_path = os.path.join(process_path, filename)
    with open(file_path, "wb") as f:
        f.write(decoded)
    reproj_file_path = os.path.join(
        process_path, filename.split(".")[0] + "_reproj.tif"
    )
    reproj_bounds, img_byte_arr, png_path = reproject_to_epsg4326(
        file_path, reproj_file_path
    )
    leaflet_bounds = [
        [reproj_bounds[1], reproj_bounds[0]],
        [reproj_bounds[3], reproj_bounds[2]],
    ]

    map_center = [
        (leaflet_bounds[0][0] + leaflet_bounds[1][0]) / 2,
        (leaflet_bounds[0][1] + leaflet_bounds[1][1]) / 2,
    ]
    encoded_img = base64.b64encode(img_byte_arr).decode("ascii")
    encoded_img = "{}{}".format("data:image/png;base64, ", encoded_img)
    print("also here")
    list_children = dl.Map(
        [
            dl.LayersControl(
                [
                    dl.Overlay(dl.TileLayer(), name="OSM", checked=True),
                    dl.Overlay(wms_layer, name="Sentinel-2", checked=True),
                    dl.Overlay(esri_tile_layer, name="ESRI", checked=True),
                ],
                id="layers-control",
                collapsed=True,
            ),
            dl.ImageOverlay(opacity=1, url=encoded_img, bounds=leaflet_bounds),
            dl.FeatureGroup(
                [
                    dl.LocateControl(
                        options={"locateOptions": {"enableHighAccuracy": True}}
                    ),
                    dl.MeasureControl(
                        position="topleft",
                        primaryLengthUnit="kilometers",
                        primaryAreaUnit="sqmeters",
                        id="measure_control",
                    ),
                    dl.EditControl(
                        id="edit_control",
                        draw={
                            "polyline": False,
                            "polygon": False,
                            "circle": False,
                            "circlemarker": False,
                        },
                    ),
                ]
            ),
        ],
        id="map",
        style={
            "width": "100%",
            "height": "80vh",
            "margin": "auto",
            "display": "block",
        },
        center=map_center,
        zoom=12,
        bounds=leaflet_bounds,
    )
    return [[reproj_file_path, png_path, leaflet_bounds]], list_children


@app.callback(
    Output("prev-table-data", "data"),
    [Input("edit_control", "geojson"), State("annotations-table", "data")],
)
def get_polygons(geojson_data, prev_data):
    if not geojson_data:
        raise PreventUpdate
    if not geojson_data["features"]:
        raise PreventUpdate
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    gdf.set_geometry = gdf["geometry"]  # type: ignore
    annotations_table_data = [shape_to_table_row(gdf.loc[[i]]) for i in range(len(gdf))]
    annotations_table_data = [
        (
            {**annotations_table_data[i], "type": "Object BBox"}
            if len(annotations_table_data[i].keys()) == 5
            else {**annotations_table_data[i], "type": "Foreground Point"}
        )
        for i in range(len(annotations_table_data))
    ]
    if list(set(gdf["type"])) == ["marker"]:
        gdf["geometry"] = gdf.geometry.buffer(0.05, cap_style=3)
        # gdf.set_geometry = gdf["geometry"]
    gdf_bounds = gdf.total_bounds
    gdf_bounds = shapely.geometry.box(*gdf_bounds).buffer(0.005).bounds
    gdf_bounds_geom = shapely.box(*gdf_bounds)
    bounds_row = bounds_to_table_row(gdf_bounds_geom)
    bounds_row["type"] = "ROI BBox"
    annotations_table_data.append(bounds_row)
    if prev_data is not None:
        prev_data = [d for d in prev_data if d["type"] != "ROI BBox"]

        intersection = [
            d1["id"]
            for d1 in prev_data
            for d2 in annotations_table_data
            if d1["id"] == d2["id"]
        ]
        annotations_table_data = [
            d2 for d2 in annotations_table_data if d2["id"] not in intersection
        ]

        annotations_table_data = annotations_table_data + prev_data

    return annotations_table_data


@app.callback(
    [
        Output("annotations-table", "data"),
    ],
    Input("prev-table-data", "data"),
)
def update_table(prev_data):
    if not prev_data:
        raise PreventUpdate
    return [prev_data]


@app.callback(
    [
        Output("downloaded_image_path", "data"),
        Output("segment-button", "disabled"),
        Output("map-card", "children", allow_duplicate=True),
        Output("download-button", "disabled"),
    ],
    [
        Input("annotations-table", "data"),
        Input("segment-button", "n_clicks"),
        Input("sam-model", "value"),
        Input("pred-iou-thresh", "value"),
        Input("stability-score-thresh", "value"),
        Input("output-data-upload", "data"),
    ],
    prevent_initial_call="initial_duplicate",
)
def run_segmentation(
    table_data,
    n_clicks,
    sam_model,
    pred_iou_thresh,
    stability_score_thresh,
    input_data,
):
    if n_clicks == 0 or table_data is None:
        raise PreventUpdate
    if n_clicks == 1 and table_data is not None:
        roi = [row for row in table_data if row["type"] == "ROI BBox"][0]
        if input_data:
            image_bounds = input_data[0][2]
            print(input_data)
            tmp_img_path = input_data[0][0]
            roi_bbox = [
                float(image_bounds[0][1]),  # x_min (longitude of SW)
                float(image_bounds[0][0]),  # y_min (latitude of SW)
                float(image_bounds[1][1]),  # x_max (longitude of NE)
                float(image_bounds[1][0]),  # y_max (latitude of NE)
            ]
            map_center = [
                (image_bounds[0][0] + image_bounds[1][0]) / 2,
                (image_bounds[0][1] + image_bounds[1][1]) / 2,
            ]
        else:
            roi_bbox = [
                float(roi["x_min"]),
                float(roi["y_min"]),
                float(roi["x_max"]),
                float(roi["y_max"]),
            ]
            image_bounds = [[roi_bbox[1], roi_bbox[0]], [roi_bbox[3], roi_bbox[2]]]
            map_center = [
                (float(roi["y_min"]) + float(roi["y_max"])) / 2.0,
                (float(roi["x_min"]) + float(roi["x_max"])) / 2.0,
            ]
            tmp_img_path = download_from_wms(
                WMS_URL, roi_bbox, LAYER, IMAGE_FORMAT, WORK_DIR, RESOLUTION  # type: ignore
            )
        types = [row["type"] for row in table_data]
        unique_types = list(set(types))
        if len(table_data) == 2 and unique_types == ["ROI BBox"]:
            segmetnation_path, png_path = generate_automatic_mask(
                tmp_img_path, sam_model, pred_iou_thresh, stability_score_thresh
            )

        else:
            bboxes_geo, foreground_points, background_points, stacked_points, labels = (
                None,
                None,
                None,
                None,
                None,
            )
            geom_df = pd.DataFrame(table_data)
            unique_types = list(set(unique_types) - set("ROI BBox"))
            for u_t in unique_types:
                tmp_df = geom_df.loc[geom_df["type"] == u_t]
                if u_t == "Object BBox":
                    bboxes_geo = tmp_df[columns[2:]].astype(float).values
                if u_t == "Foreground Point":
                    foreground_points = tmp_df[columns[2:4]].astype(float).values
                if u_t == "Background Point":
                    background_points = tmp_df[columns[2:4]].astype(float).values
            if foreground_points is None and background_points is not None:
                stacked_points = np.copy(background_points)
                labels = np.zeros((stacked_points.shape[0]), dtype=np.uint8)
            elif background_points is None and foreground_points is not None:
                stacked_points = np.copy(foreground_points)
                labels = np.ones((stacked_points.shape[0]), dtype=np.uint8)
            elif background_points is not None and foreground_points is not None:
                stacked_points = np.vstack((background_points, foreground_points))
                labels = np.zeros((stacked_points.shape[0]), dtype=np.uint8)
                labels[background_points.shape[0] :] = 1

            segmetnation_path, png_path = sam_prompt_bbox(
                tmp_img_path, bboxes_geo, stacked_points, labels, sam_model, roi_bbox
            )
        encoded_img = base64.b64encode(open(png_path, "rb").read()).decode("ascii")
        encoded_img = "{}{}".format("data:image/png;base64, ", encoded_img)
        list_children_items = [
            dl.LayersControl(
                [
                    dl.Overlay(dl.TileLayer(), name="OSM", checked=True),
                    dl.Overlay(wms_layer, name="Sentinel-2", checked=True),
                    dl.Overlay(esri_tile_layer, name="ESRI", checked=True),
                ],
                id="layers-control",
                collapsed=True,
            ),
            dl.ImageOverlay(opacity=0.5, url=encoded_img, bounds=image_bounds),
            dl.FeatureGroup(
                [
                    dl.LocateControl(
                        options={"locateOptions": {"enableHighAccuracy": True}}
                    ),
                    dl.MeasureControl(
                        position="topleft",
                        primaryLengthUnit="kilometers",
                        primaryAreaUnit="sqmeters",
                        id="measure_control",
                    ),
                    dl.EditControl(
                        id="edit_control",
                        draw={
                            "polyline": False,
                            "polygon": False,
                            "circle": False,
                            "circlemarker": False,
                        },
                    ),
                ]
            ),
        ]
        if input_data:
            encoded_img = base64.b64encode(open(input_data[0][1], "rb").read()).decode(
                "ascii"
            )
            encoded_img = "{}{}".format("data:image/png;base64, ", encoded_img)
            list_children_items.insert(
                0, dl.ImageOverlay(opacity=0.5, url=encoded_img, bounds=image_bounds)
            )

        list_children = dl.Map(
            list_children_items,
            id="map",
            style={
                "width": "100%",
                "height": "80vh",
                "margin": "auto",
                "display": "block",
            },
            center=map_center,
            zoom=15,
            bounds=image_bounds,
        )
        return [tmp_img_path, segmetnation_path, png_path], True, list_children, False


@app.callback(
    Output("download-link", "data"),
    [Input("download-button", "n_clicks"), Input("downloaded_image_path", "data")],
)
def prepare_downloadble(n_clicks, download_data):
    if n_clicks == 0 or download_data is None:
        raise PreventUpdate

    def write_archive(memory_file):
        # memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "a", zipfile.ZIP_DEFLATED) as z_file:
            for file_path in download_data:
                z_file.write(file_path)
        memory_file.seek(0)

    return dcc.send_bytes(
        write_archive,
        os.path.basename(download_data[0]).split(".")[0] + ".zip",
    )


if __name__ == "__main__":
    # app.run_server(debug=DEBUG, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run_server(debug=DEBUG, port=8062)
