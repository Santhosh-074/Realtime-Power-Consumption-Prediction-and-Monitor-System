(function () {
  const cfg = window.RTPCPMS;
  const id = cfg.datasetId;
  const BLUE = '#0f6cbd', ORANGE = '#f59e0b', RED = '#dc3545', GREY = '#6c757d';

  function thresholdLine(n, value) {
    return value == null ? null : {
      label: 'Threshold', data: Array(n).fill(value), borderColor: RED, borderDash: [6, 4],
      borderWidth: 1.5, pointRadius: 0, fill: false
    };
  }

  function baseOptions(yTitle) {
    return {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { title: { display: true, text: yTitle }, beginAtZero: true }
      }
    };
  }

  // ---- Historical chart (F-015) ----
  let historyChart = null;
  async function loadHistory() {
    const start = document.getElementById('startDate').value;
    const end = document.getElementById('endDate').value;
    const res = await fetch(`/api/datasets/${id}/history?start=${start}&end=${end}`);
    const data = await res.json();
    const datasets = [{
      label: 'Power (kW)', data: data.values, borderColor: BLUE, backgroundColor: 'rgba(15,108,189,.1)',
      borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.15
    }];
    const t = thresholdLine(data.values.length, data.threshold);
    if (t) datasets.push(t);
    if (historyChart) historyChart.destroy();
    historyChart = new Chart(document.getElementById('historyChart'), {
      type: 'line', data: { labels: data.labels, datasets }, options: baseOptions('kW')
    });
    const note = document.getElementById('bucketNote');
    const names = { '1h': 'hourly', '3h': '3-hour', '6h': '6-hour', '12h': '12-hour', D: 'daily', W: 'weekly', MS: 'monthly' };
    note.textContent = data.bucket && data.bucket !== 'raw'
      ? `Large range: showing ${names[data.bucket]} averages to keep the chart readable. Narrow the dates to see detail.`
      : '';
  }
  document.getElementById('rangeForm').addEventListener('submit', (e) => { e.preventDefault(); loadHistory(); });
  loadHistory();

  // ---- Forecast chart (F-016) ----
  if (cfg.hasPredictions) {
    fetch(`/api/datasets/${id}/prediction`).then(r => r.json()).then(data => {
      const h = data.history, f = data.forecast;
      const labels = h.labels.concat(f.labels);
      const actual = h.values.concat(Array(f.values.length).fill(null));
      // start the forecast line at the last actual point so the two lines connect
      const pred = Array(h.values.length - 1).fill(null)
        .concat([h.values[h.values.length - 1]]).concat(f.values);
      const datasets = [
        { label: 'Actual (recent)', data: actual, borderColor: BLUE, borderWidth: 2, pointRadius: 0, tension: 0.15 },
        { label: 'Predicted', data: pred, borderColor: ORANGE, borderWidth: 2.5, pointRadius: 2, borderDash: [5, 3], tension: 0.15 }
      ];
      const t = thresholdLine(labels.length, data.threshold);
      if (t) datasets.push(t);
      new Chart(document.getElementById('predChart'), {
        type: 'line', data: { labels, datasets }, options: baseOptions('kW')
      });
    });
  }

  // ---- Model evaluation chart (F-013) ----
  if (cfg.hasModel) {
    fetch(`/api/datasets/${id}/evaluation`).then(r => r.json()).then(data => {
      new Chart(document.getElementById('evalChart'), {
        type: 'line',
        data: {
          labels: data.labels,
          datasets: [
            { label: 'Actual', data: data.actual, borderColor: BLUE, borderWidth: 1.5, pointRadius: 0 },
            { label: 'Model prediction', data: data.predicted, borderColor: ORANGE, borderWidth: 1.5, pointRadius: 0 }
          ]
        },
        options: baseOptions('kW')
      });
    });
  }
})();
