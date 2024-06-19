import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

path = os.path.expanduser('~/airflow_hw')
os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 2, 21),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    first_task = BashOperator(
        task_id='first_task',
        bash_command='echo "Pipeline is starting!"',
        dag=dag,
    )

    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag,
    )

    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag,
    )

    first_task >> pipeline >> predict


