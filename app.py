from flask import Flask, request, render_template


app = Flask(__name__)


@app.route('/')
def index():
    return 'hello'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)