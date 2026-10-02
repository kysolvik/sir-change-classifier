#!/usr/bin/env bash
# Deploy the classifier to Cloud Run with bounded cost. Idempotent: re-running
# skips anything that already exists and just ships a new revision.
#
#   gcloud auth login && gcloud auth application-default login   # once
#   PROJECT=solvik-montero-sir ./scripts/deploy.sh
#
# Cost guard rails (see README "Cost & abuse limits"):
#   * scale to zero + request-based billing: pay only while requests run
#   * MAX_INSTANCES caps worst-case compute (~$139/month per instance pegged 24/7)
#   * per-IP limits + a global hourly classify cap inside the app
#   * a billing budget with early email alerts + a request-flood alert
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT=<gcp project id>}
REGION=${REGION:-us-east1}
SERVICE=${SERVICE:-sir-classifier}
BUCKET=${BUCKET:-${PROJECT}-aef-cache}
SA_NAME=${SA_NAME:-sir-classifier}
MAX_INSTANCES=${MAX_INSTANCES:-3}
BUDGET_USD=${BUDGET_USD:-50}
ALERT_EMAIL=${ALERT_EMAIL:-$(gcloud config get-value account 2>/dev/null)}
SKIP_ALERTS=${SKIP_ALERTS:-}  # set to 1 on redeploys to skip budget/alert setup

SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
g() { gcloud --project "$PROJECT" "$@"; }

echo "== APIs"
g services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com \
  billingbudgets.googleapis.com monitoring.googleapis.com

echo "== Cache bucket gs://${BUCKET}"
if ! g storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  g storage buckets create "gs://${BUCKET}" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi

echo "== Service account ${SA}"
if ! g iam service-accounts describe "$SA" >/dev/null 2>&1; then
  g iam service-accounts create "$SA_NAME" --display-name "SIR classifier (Cloud Run)"
fi
# Only permission it has: read/write cached windows in its own bucket.
g storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SA}" --role roles/storage.objectAdmin >/dev/null

echo "== Build & deploy ${SERVICE}"
# --concurrency 8: RF prediction already uses both vCPUs, so extra concurrency
# adds no throughput, but lets a classroom burst queue inside the instance
# instead of being rejected (3 x 8 = 24 slots for ~20 students).
g run deploy "$SERVICE" --source . --region "$REGION" \
  --service-account "$SA" --allow-unauthenticated \
  --cpu 2 --memory 2Gi --timeout 300 --concurrency 8 \
  --min-instances 0 --max-instances "$MAX_INSTANCES" --cpu-throttling \
  --set-env-vars "GCS_CACHE_BUCKET=${BUCKET},INPROC_CACHE_SIZE=4,WEB_CONCURRENCY=2"

[[ -n "$SKIP_ALERTS" ]] && exit 0

echo "== Billing budget (\$${BUDGET_USD}/month)"
BILLING=$(g billing projects describe "$PROJECT" --format 'value(billingAccountName)')
BILLING=${BILLING#billingAccounts/}
if gcloud billing budgets list --billing-account "$BILLING" \
     --format 'value(displayName)' 2>/dev/null | grep -qx "$SERVICE"; then
  echo "budget exists"
else
  # Emails billing admins at each threshold; 10% (= $5) is the early warning.
  gcloud billing budgets create --billing-account "$BILLING" \
    --display-name "$SERVICE" --budget-amount "${BUDGET_USD}USD" \
    --filter-projects "projects/${PROJECT}" \
    --threshold-rule percent=0.1 --threshold-rule percent=0.25 \
    --threshold-rule percent=0.5 --threshold-rule percent=0.9 \
    --threshold-rule percent=1.0 \
    --threshold-rule percent=1.0,basis=forecasted-spend \
  || echo "!! Budget creation failed (need Billing Account Administrator?). Create it in the console."
fi

echo "== Request-flood alert -> ${ALERT_EMAIL}"
# Billing data lags up to a day; this fires within minutes of a flood.
if g alpha monitoring policies list --format 'value(displayName)' 2>/dev/null \
     | grep -qx "${SERVICE} request flood"; then
  echo "alert exists"
else
  CHANNEL=$(g beta monitoring channels list \
    --filter "type=\"email\" AND labels.email_address=\"${ALERT_EMAIL}\"" \
    --format 'value(name)' | head -n1)
  if [[ -z "$CHANNEL" ]]; then
    CHANNEL=$(g beta monitoring channels create --type email \
      --display-name "$ALERT_EMAIL" --channel-labels "email_address=${ALERT_EMAIL}" \
      --format 'value(name)')
  fi
  POLICY=$(mktemp)
  cat >"$POLICY" <<EOF
{
  "displayName": "${SERVICE} request flood",
  "combiner": "OR",
  "conditions": [{
    "displayName": "> 5000 requests in 5 min",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE}\" AND metric.type=\"run.googleapis.com/request_count\"",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM",
                        "crossSeriesReducer": "REDUCE_SUM"}],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 5000,
      "duration": "0s"
    }
  }],
  "notificationChannels": ["${CHANNEL}"]
}
EOF
  g alpha monitoring policies create --policy-from-file "$POLICY"
  rm -f "$POLICY"
fi

URL=$(g run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: ${URL}"
echo "Warm the preset cache once:  GCS_CACHE_BUCKET=${BUCKET} uv run python -m scripts.warm_cache"
echo "Emergency brake:  gcloud run services update ${SERVICE} --region ${REGION} --project ${PROJECT} --ingress internal"
