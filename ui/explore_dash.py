import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import json
import glob
import os
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from flask_caching import Cache

nltk.download('punkt')

from utils import get_jsonl_files, group_data, analyze_text_cleanliness, analyze_corpus_cleanliness, load_jsonl_gz

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 300  # Cache timeout in seconds

def get_subfolders(base_dir):
    return [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

@cache.memoize(timeout=TIMEOUT)
def get_data_structure(base_dir):
    structure = {'datasets': [], 'output': [], 'filters': defaultdict(list)}
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if item == 'output':
                # Get all .jsonl.gz files in the output directory
                output_files = [f for f in os.listdir(item_path) if f.endswith('.jsonl.gz')]
                # Extract unique dataset names from file names
                structure['output'] = list(set(f.split('_')[0] for f in output_files))
                # Add 'All Outputs' option if there are any output files
                if output_files:
                    structure['output'].insert(0, 'All Outputs')
            else:
                structure['datasets'].append(item)
                for filter_name in os.listdir(item_path):
                    if os.path.isdir(os.path.join(item_path, filter_name)):
                        structure['filters'][item].append(filter_name)
    return structure

@cache.memoize(timeout=TIMEOUT)
def get_data(base_dir, data_type, dataset):
    if data_type == 'output':
        path = os.path.join(base_dir, 'output')
        if dataset == 'All Outputs':
            files = [f for f in os.listdir(path) if f.endswith('.jsonl.gz')]
        else:
            files = [f for f in os.listdir(path) if f.startswith(f"{dataset}_") and f.endswith('.jsonl.gz')]
        data = []
        for file in files:
            data.extend(load_jsonl_gz(os.path.join(path, file)))
    else:  # filtered data
        path = os.path.join(base_dir, dataset)
        grouped_data = group_data(path)
        data = [doc for group in grouped_data.values() for doc in group]
    return data

@cache.memoize(timeout=TIMEOUT)
def get_corpus_stats(base_dir, data_type, dataset):
    data = get_data(base_dir, data_type, dataset)
    if not data:
        return None
    cleanliness_metrics = [analyze_text_cleanliness(doc['text']) for doc in data]
    return {
        'avg_word_count': sum(m['word_count'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_sentence_count': sum(m['sentence_count'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_word_length': sum(m['avg_word_length'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_special_char_ratio': sum(m['special_char_ratio'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_uppercase_ratio': sum(m['uppercase_ratio'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_newline_count': sum(m['newline_count'] for m in cleanliness_metrics) / len(cleanliness_metrics)
    }

@cache.memoize(timeout=TIMEOUT)
def get_cleanliness_metrics(base_dir, data_type, dataset):
    data = get_data(base_dir, data_type, dataset)
    return [analyze_text_cleanliness(doc['text']) for doc in data]

@cache.memoize(timeout=TIMEOUT)
def get_sample_previews(base_dir, data_type, dataset, n=10):
    data = get_data(base_dir, data_type, dataset)
    samples = pd.DataFrame(data).sample(n=min(n, len(data)))
    return [
        {
            'id': row.get('id', 'N/A'),
            'preview': row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
            'metadata': {k: v for k, v in row.items() if k != 'text'}
        }
        for _, row in samples.iterrows()
    ]

@cache.memoize(timeout=TIMEOUT)
def get_document(base_dir, data_type, dataset, doc_id):
    data = get_data(base_dir, data_type, dataset)
    return next((d for d in data if d.get('id') == doc_id), None)

def get_rejection_reasons(base_dir, dataset, doc_id):
    structure = get_data_structure(base_dir)
    filters = structure['filters'].get(dataset, [])

    for filter_name in filters:
        filter_path = os.path.join(base_dir, dataset, filter_name)
        if os.path.isdir(filter_path):
            # Find all JSONL.gz files in the filter directory
            jsonl_files = glob.glob(os.path.join(filter_path, '*.jsonl.gz'))
            for jsonl_file in jsonl_files:
                try:
                    filter_data = load_jsonl_gz(jsonl_file)
                    if not any(doc['id'] == doc_id for doc in filter_data):
                        return [filter_name]
                except Exception as e:
                    print(f"Error loading {jsonl_file}: {str(e)}")

    # If the document is not found in any filter, it passed all filters
    return []

app.layout = dbc.Container([
    html.H1("Text Processing Pipeline Explorer"),

    dbc.Row([
        dbc.Col([
            html.H3("Base Data Directory"),
            dcc.Input(id='base-dir-input', type='text', placeholder="Enter base data directory path", style={'width': '100%'}),
        ], width=3),
        dbc.Col([
            html.H3("Data Type"),
            dcc.Dropdown(
                id='data-type-dropdown',
                options=[
                    {'label': 'Filtered Data', 'value': 'filtered'},
                    {'label': 'Output Data', 'value': 'output'}
                ],
                value=None,
                placeholder="Select data type"
            )
        ], width=3),
        dbc.Col([
            html.H3("Dataset"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[],
                value=None,
                placeholder="Select a dataset"
            )
        ], width=3),
        dbc.Col([
            html.H3("Analysis Type"),
            dcc.Dropdown(
                id='analysis-type-dropdown',
                options=[
                    {'label': 'Corpus Overview', 'value': 'overview'},
                    {'label': 'Cleanliness Metrics', 'value': 'cleanliness'},
                    {'label': 'Text Samples', 'value': 'samples'}
                ],
                value='overview'
            )
        ], width=3)
    ]),

    html.Div(id='analysis-output'),

    dbc.Button("Load More Samples", id="load-more-samples", color="secondary", className="mt-3"),

    dbc.Modal([
        dbc.ModalHeader(id="sample-modal-header"),
        dbc.ModalBody([
            dbc.Tabs([
                dbc.Tab(label="Metadata", tab_id="metadata"),
                dbc.Tab(label="Full Text", tab_id="full-text"),
            ], id="sample-tabs", active_tab="metadata"),
            html.Div(id="sample-tab-content")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-sample-modal", className="ml-auto")
        ),
    ], id="sample-modal", size="lg"),
])

@app.callback(
    Output('data-type-dropdown', 'options'),
    Output('data-type-dropdown', 'value'),
    Input('base-dir-input', 'value')
)
def update_data_type_options(base_dir):
    if not base_dir:
        return [], None
    structure = get_data_structure(base_dir)
    options = []
    if structure['datasets']:
        options.append({'label': 'Filtered Data', 'value': 'filtered'})
    if structure['output']:
        options.append({'label': 'Output Data', 'value': 'output'})
    return options, options[0]['value'] if options else None

def generate_overview(base_dir, data_type, dataset):
    data = get_data(base_dir, data_type, dataset)
    if not data:
        return html.Div([
            html.H2(f"Overview of {dataset} ({data_type})"),
            html.P("No data available for this dataset.")
        ])

    stats = get_corpus_stats(base_dir, data_type, dataset)

    # Calculate additional metrics
    num_documents = len(data)
    total_words = sum(len(doc['text'].split()) for doc in data)
    avg_doc_length = total_words / num_documents if num_documents > 0 else 0
    unique_words = len(set(word.lower() for doc in data for word in doc['text'].split()))

    # Get the most common words (excluding common stop words)
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for doc in data for word in doc['text'].split() if word.lower() not in stop_words]
    common_words = Counter(words).most_common(10)

    return html.Div([
        html.H2(f"Overview of {dataset} ({data_type})"),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("Number of Documents"), html.Td(f"{num_documents:,}")]),
            html.Tr([html.Td("Total Word Count"), html.Td(f"{total_words:,}")]),
            html.Tr([html.Td("Unique Words"), html.Td(f"{unique_words:,}")]),
            html.Tr([html.Td("Average Document Length (words)"), html.Td(f"{avg_doc_length:.2f}")]),
            html.Tr([html.Td("Average Word Count per Document"), html.Td(f"{stats['avg_word_count']:.2f}")]),
            html.Tr([html.Td("Average Sentence Count per Document"), html.Td(f"{stats['avg_sentence_count']:.2f}")]),
            html.Tr([html.Td("Average Word Length"), html.Td(f"{stats['avg_word_length']:.2f}")]),
            html.Tr([html.Td("Average Special Character Ratio"), html.Td(f"{stats['avg_special_char_ratio']:.4f}")]),
            html.Tr([html.Td("Average Uppercase Ratio"), html.Td(f"{stats['avg_uppercase_ratio']:.4f}")]),
            html.Tr([html.Td("Average Newline Count per Document"), html.Td(f"{stats['avg_newline_count']:.2f}")]),
        ], className="table table-striped table-bordered"),

        html.H3("Most Common Words (excluding stop words)"),
        html.Ol([html.Li(f"{word}: {count}") for word, count in common_words])
    ])

def generate_cleanliness_analysis(base_dir, data_type, dataset):
    metrics = get_cleanliness_metrics(base_dir, data_type, dataset)
    df = pd.DataFrame(metrics)

    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Box(y=df[column], name=column))

    fig.update_layout(title=f"Cleanliness Metrics Distribution for {dataset} ({data_type})")

    return html.Div([
        dcc.Graph(figure=fig),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Average"), html.Th("Median"), html.Th("Min"), html.Th("Max")]),
            *[html.Tr([
                html.Td(column),
                html.Td(f"{df[column].mean():.2f}"),
                html.Td(f"{df[column].median():.2f}"),
                html.Td(f"{df[column].min():.2f}"),
                html.Td(f"{df[column].max():.2f}")
            ]) for column in df.columns]
        ], className="table table-striped table-bordered")
    ])

@app.callback(
    Output('analysis-output', 'children'),
    Input('base-dir-input', 'value'),
    Input('data-type-dropdown', 'value'),
    Input('dataset-dropdown', 'value'),
    Input('analysis-type-dropdown', 'value'),
    Input('load-more-samples', 'n_clicks')
)
def update_analysis(base_dir, data_type, dataset, analysis_type, n_clicks):
    if not base_dir or not data_type or not dataset or not analysis_type:
        return "Please enter a base directory, select a data type, dataset, and analysis type."

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if analysis_type == 'overview':
        return generate_overview(base_dir, data_type, dataset)
    elif analysis_type == 'cleanliness':
        return generate_cleanliness_analysis(base_dir, data_type, dataset)
    elif analysis_type == 'samples':
        n = 10 if triggered_id != 'load-more-samples' else (n_clicks + 1) * 10
        samples = get_sample_previews(base_dir, data_type, dataset, n)
        return generate_text_samples(base_dir, data_type, dataset, samples)

def generate_text_samples(base_dir, data_type, dataset, samples):
    return html.Div([
        dbc.Card([
            dbc.CardHeader(f"Document ID: {sample['id']}"),
            dbc.CardBody([
                html.P(sample['preview']),
                html.P(f"Rejected by: {', '.join(get_rejection_reasons(base_dir, dataset, sample['id']))}" if data_type == 'filtered' else "Passed all filters"),
                dbc.Button("View Details", id={'type': 'view-sample', 'index': sample['id']}, color="primary")
            ])
        ], className="mb-3")
        for sample in samples
    ])

@app.callback(
    Output('load-more-samples', 'style'),
    Input('analysis-type-dropdown', 'value')
)
def toggle_load_more_button(analysis_type):
    if analysis_type == 'samples':
        return {'display': 'block'}
    return {'display': 'none'}

from dash.exceptions import PreventUpdate

from dash.exceptions import PreventUpdate

@app.callback(
    Output("sample-modal", "is_open"),
    Output("sample-modal-header", "children"),
    Output("sample-tab-content", "children"),
    Output("sample-tabs", "active_tab"),
    Input({'type': 'view-sample', 'index': ALL}, 'n_clicks'),
    Input("close-sample-modal", "n_clicks"),
    Input("sample-tabs", "active_tab"),
    State("sample-modal", "is_open"),
    State("sample-modal-header", "children"),
    State('base-dir-input', 'value'),
    State('data-type-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
)
def update_modal(view_clicks, close_clicks, active_tab, is_open, current_header, base_dir, data_type, dataset):
    ctx = dash.callback_context
    print(f"Callback triggered. Triggered ID: {ctx.triggered_id}")

    if not ctx.triggered:
        return is_open, current_header, dash.no_update, active_tab

    triggered_id = ctx.triggered_id

    if triggered_id == "close-sample-modal":
        return False, "", dash.no_update, "metadata"

    if isinstance(triggered_id, dict) and triggered_id.get('type') == 'view-sample':
        doc_id = triggered_id['index']
        print(f"View sample clicked for doc_id: {doc_id}")
    elif triggered_id == "sample-tabs":
        doc_id = current_header.split(": ")[1] if ": " in current_header else None
        print(f"Tab changed. Doc ID from header: {doc_id}")
    else:
        return is_open, current_header, dash.no_update, active_tab

    if doc_id is None:
        return is_open, current_header, dash.no_update, active_tab

    print(f"Fetching document: base_dir={base_dir}, data_type={data_type}, dataset={dataset}, doc_id={doc_id}")
    doc = get_document(base_dir, data_type, dataset, doc_id)

    if doc:
        print(f"Document found: {doc.get('id', 'N/A')}")
        header = f"Document ID: {doc.get('id', 'N/A')}"
        if active_tab == "metadata":
            content = generate_metadata_view(doc)
        else:
            content = generate_full_text_view(doc)
        print(f"Returning - is_open: True, header: {header}, content type: {type(content)}, active_tab: {active_tab}")
        return True, header, content, active_tab
    else:
        print("Document not found")
        return True, "Document Not Found", "The requested document could not be found.", active_tab

@app.callback(
    Output('dataset-dropdown', 'options'),
    Output('dataset-dropdown', 'value'),
    Input('base-dir-input', 'value'),
    Input('data-type-dropdown', 'value')
)
def update_dataset_options(base_dir, data_type):
    if not base_dir or not data_type:
        return [], None

    structure = get_data_structure(base_dir)

    if data_type == 'filtered':
        options = [{'label': dataset, 'value': dataset} for dataset in structure['datasets']]
    else:  # output
        options = [{'label': dataset, 'value': dataset} for dataset in structure['output']]

    return options, options[0]['value'] if options else None

def generate_metadata_view(doc):
    cleanliness_metrics = analyze_text_cleanliness(doc['text'])
    metadata = {
        "Word Count": cleanliness_metrics['word_count'],
        "Sentence Count": cleanliness_metrics['sentence_count'],
        "Average Word Length": f"{cleanliness_metrics['avg_word_length']:.2f}",
        "Special Character Ratio": f"{cleanliness_metrics['special_char_ratio']:.4f}",
        "Uppercase Ratio": f"{cleanliness_metrics['uppercase_ratio']:.4f}",
        "Newline Count": cleanliness_metrics['newline_count']
    }

    # Function to format value
    def format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        elif isinstance(value, list):
            return ", ".join(map(str, value))
        elif value is None:
            return "None"
        else:
            return str(value)

    # Unpack and add other metadata
    for k, v in doc.items():
        if k != 'text':
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    metadata[f"{k}_{sub_k}"] = format_value(sub_v)
            elif isinstance(v, list) and all(isinstance(item, dict) for item in v):
                for i, item in enumerate(v):
                    for sub_k, sub_v in item.items():
                        metadata[f"{k}_{i+1}_{sub_k}"] = format_value(sub_v)
            else:
                metadata[k] = format_value(v)

    # Create table rows
    rows = []
    for k, v in metadata.items():
        rows.append(html.Tr([
            html.Th(k, style={'width': '30%'}),
            html.Td(v, style={'width': '70%', 'word-break': 'break-word'})
        ]))

    return html.Table(rows, className="table table-striped table-bordered")

def generate_full_text_view(doc):
    return html.Div([
        html.Pre(
            doc['text'],
            style={
                'white-space': 'pre-wrap',
                'word-break': 'break-word',
                'font-family': 'monospace',
                'font-size': '14px',
                'line-height': '1.5',
                'padding': '10px',
                'border': '1px solid #ddd',
                'border-radius': '4px',
                'background-color': '#f8f9fa'
            }
        )
    ], style={'max-height': '500px', 'overflow-y': 'auto'})

if __name__ == '__main__':
    app.run_server(debug=True)
