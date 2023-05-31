import base64
from io import BytesIO
import json 

from flask import Flask, request 
from matplotlib.figure import Figure
from flask_cors import CORS, cross_origin

from VQE import VQE_molecule 

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/vqe', methods=['POST'])
def vqe():
	print(request.data)
	data = json.loads(request.data.decode('utf8').replace("'", '"'))
	fig = Figure()
	ax = fig.subplots()
	ax.plot([1, 2])
	buf = BytesIO()
	fig.savefig(buf, format="png")
	result = VQE_molecule(molecule=data.get("molecule"), distance=1.59, noise=False, mitigated=False)
	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	return data 
	# return f"<img src='data:image/png;base64,{data}'/>"
if __name__ == '__main__':
	app.run(debug=True)
	