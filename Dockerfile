FROM docker.repo.local.sfdc.net/sfci/docker-images/sfdc_rhel9_python3:120

LABEL maintainer interactive-analytics@salesforce.com

ENV WORKDIR=/app
WORKDIR ${WORKDIR}
#s3 path is configured via trino-gateway as env variable.
ENV QUERY_PREDICTOR_S3_BUCKET=placeholder

#Install aws cli bundle and build dependencies
RUN add_strata_repo.py --name image-dependencies --type rpm-rc-signed-v2 \
    && add_strata_repo.py --name thirdparty_docker-ce --type rpm-rc-signed-v2 \
    && dnf -y install \
    awscli-bundle \
    tini \
    cronie \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    libgomp \
    && dnf clean all

COPY . trino-query-predictor

RUN pip install -e trino-query-predictor/

ENV PYTHONPATH "${PYTHONPATH}:/app/trino-query-predictor"

EXPOSE 8000

ENV SCRIPT_WORKDIR /app/trino-query-predictor/scripts
WORKDIR ${SCRIPT_WORKDIR}
RUN chmod 755 ${SCRIPT_WORKDIR}/*sh
RUN chmod -R 777 /app /var/run

# Health check for Flask service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/manage/health/liveness || exit 1

# Run the Flask service via Waitress
CMD ["python3", "-m", "query_predictor.service"]

LABEL version="1.0"

RUN useradd -u 7447 --create-home --shell /bin/bash cronutil
RUN chown -R cronutil:cronutil  /home/cronutil

# Set permissions for crontab (setuid).
RUN chmod gu+s /usr/sbin/crond

USER root

WORKDIR ${WORKDIR}
