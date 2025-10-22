#!/bin/bash
crond

su - cronutil <<'EOF'

QUERY_PREDICTOR_S3_BUCKET="$1"
source /app/trino-query-predictor/examples/load-aws-credential.sh
# TODO: Uncomment this when we we are ready to fetch from s3
# echo "* * * * * /app/trino-query-predictor/scripts/fetch-from-s3.sh $QUERY_PREDICTOR_S3_BUCKET" | crontab -

# Start the query-predictor application
python3 /app/trino-query-predictor/query_predictor/service/app.py &

EOF

# Start the cron daemon to fetch from s3
#crond