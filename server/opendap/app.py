import os
from pydap.wsgi.app import DapServer

data_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "opendap", "data_root")

app = DapServer(data_dir)