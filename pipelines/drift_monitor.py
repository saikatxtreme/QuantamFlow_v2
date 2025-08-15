import os, pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping
from quantumflow_core import load_config, read_csv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def send_slack(msg, webhook=None, token=None, channel='#general'):
    # Simple slack notifier: prefer token+chat.postMessage for richer control
    if token:
        client = WebClient(token=token)
        try:
            client.chat_postMessage(channel=channel, text=msg)
        except SlackApiError as e:
            print('Slack error', e)
    else:
        print('SLACK:', msg)

def run_drift(ref_path, curr_path, output_html='drift_report.html', slack_token=None, slack_channel='#alerts'):
    ref = pd.read_parquet(ref_path) if ref_path.endswith('.parquet') else pd.read_csv(ref_path)
    curr = pd.read_parquet(curr_path) if curr_path.endswith('.parquet') else pd.read_csv(curr_path)
    # Basic column mapping - assume same schema
    col_mapping = ColumnMapping()
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref, current_data=curr, column_mapping=col_mapping)
    report.save_html(output_html)
    send_slack(f'Drift report generated: {output_html}', token=slack_token, channel=slack_channel)
    return output_html

if __name__ == '__main__':
    cfg = load_config()
    data_dir = cfg.get('data_dir','data')
    ref = os.path.join(data_dir,'sales_enriched.parquet') if os.path.exists(os.path.join(data_dir,'sales_enriched.parquet')) else os.path.join(data_dir,'sales.csv')
    curr = os.path.join(data_dir,'sales_recent.csv') if os.path.exists(os.path.join(data_dir,'sales_recent.csv')) else ref
    run_drift(ref, curr)
