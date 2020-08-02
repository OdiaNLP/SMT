import os
import sys
from datetime import datetime

import dill as pickle
from flask import Flask, render_template, request

from form_model import InputForm

sys.path.append('src')

from smt_utils import (
    tokenize_en,
    detokenize_od,
    translate
)

# create app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return "It's working! :) "


@app.route('/translate', methods=['GET', 'POST'])
def translate():
    form = InputForm(request.form)

    if request.method == 'POST' and form.validate():

        # tokenize
        sentence = tokenize_en(form.src_text.data)

        # translate
        translation = translate(
            ibm_model=ibm_model_loaded,
            src_tokens=sentence
        )

        # detokenize
        result = detokenize_od(translation)

        """
        # display attention
        display_attention(sentence, translation, attention)
        """
    else:
        result = None

    if result is not None:
        result = f'Odia Translation: {result}'

        if responses_path is not None:
            with open(responses_path, 'a', encoding='utf-8') as _f:
                _f.write(
                    f'\n\tNEW REQUEST 🤩 @'
                    f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                    f'\tEnglish Text: {form.src_text.data}\n'
                    f'\t{result}\n'
                )

    return render_template(template_name + '.html',
                           form=form, result=result)


if __name__ == '__main__':
    # set template
    template_name = 'my_view'

    # responses dir
    responses_path = 'responses/logs.txt'

    # create responses dir
    os.makedirs('responses', exist_ok=True)

    # load model
    with open(os.path.join('models', 'model.pkl'), 'rb') as f:
        ibm_model_loaded = pickle.load(f)

    if responses_path is not None:
        with open(responses_path, 'a', encoding='utf-8') as f:
            f.write(
                f'\nstarting app.. '
                f'[{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'
                f'\n'
            )

    # run app
    app.run(host='127.0.0.1', port=31137, debug=False)
