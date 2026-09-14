"""Process-isolated notebook entry point; uses the configured job directory."""
import sys,json
from pathlib import Path
from workflow import ROOT,run_experiment,write_json
job=Path(sys.argv[1]).resolve()
request=json.loads((job/'request.json').read_text())
if job.parent!=Path(request['config'].get('jobs_dir',ROOT/'jobs')).resolve():raise ValueError('Unexpected job directory')
run=run_experiment(request['architecture'],request['config'])
write_json(job/'result.json',dict(run=str(run)))
