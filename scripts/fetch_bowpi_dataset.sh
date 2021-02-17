#!/usr/bin/env bash

NOW=$(date)
#DATASET_DATE="2021/02/02"


DAY=$(date -d "$DATASET_DATE" '+%d')
MONTH=$(date -d "$DATASET_DATE" '+%m')
YEAR=$(date -d "$DATASET_DATE" '+%Y')
TRAINING_DATA_URL="gt.cch.out.prd/v3/redshift/datascience/gt/cch/ptm/dataset/anio=$YEAR/mes=$MONTH/dia=$DAY/gt_cch_das_directSaleTuca_$YEAR$MONTH$DAY""_000"
TRAINING_DATA_PATH="packages/regression_model/regression_model/datasets/dataset.csv"

cat >> /home/ec2-user/JC/.aws/credentials << EOF
# >>> configure aws cli access >>>
[get_dataset]
aws_access_key_id = $BOWPI_USERNAME
aws_secret_access_key = $BOWPI_KEY
# <<< ends configuring aws cli access <<<
EOF

cat >> /home/ec2-user/JC/.aws/config << EOF
# >>> configure aws cli access >>>
[profile get_dataset]
region=us-east-1
output=json
# <<< ends configuring aws cli access <<<
EOF

echo $TRAINING_DATA_URL

aws s3 cp s3://$TRAINING_DATA_URL $TRAINING_DATA_PATH --profile get_dataset
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > $TRAINING_DATA_PATH
