import mimetypes
import os
import tempfile


from flask import Flask, send_file
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
from waitress import serve
from werkzeug.datastructures import FileStorage


from .gram_matrix_comparision import handle_file, handle_zip


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    api = Api(
        app,
        version='1.0.0',
        title='Style Recogniser app',
        description='Style Recogniser app',
        default='Style Recogniser app',
        default_label=''
    )

    CORS(app)

    files = reqparse.RequestParser()

    files.add_argument('File',
                        type=FileStorage,
                        location='files',
                        help='select image or zip file',
                        required=True)

    @api.route('/similar_images')
    @api.expect(files)
    class images_predict(Resource):
        @api.expect(files)
        def post(self):
            try:
                args = files.parse_args()
                file = args.get('File')
                if ".zip" in file.filename:
                    iszip = True
                else:
                    iszip = False
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                req_dir = tempfile.mkdtemp()
                fileobj = os.path.join(req_dir, file.filename)
                fileobj = open(fileobj, mode='xb')
                file.save(fileobj)
                fileobj.close()
                filepath = os.path.join(req_dir, file.filename)
                if iszip:
                    result_path = handle_zip(filepath)
                else:
                    result_path = handle_file(filepath)
                mime = mimetypes.guess_type(result_path)
                return send_file(result_path,
                                 mimetype=mime[0],
                                 attachment_filename=os.path.basename(result_path),
                                 as_attachment=True)

            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
    return app


if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=7777)
