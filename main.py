
import sys
sys.path += [
    './gpt-2/src/',
]

import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

from utils import load_settings
from reddit_news import get_reddit_news

settings = load_settings()

def run_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='gpt-2/models',
    callback=None,
):
    """
    load the saved model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :models_dir : path to parent folder containing model subfolders
    (i.e. contains the <model_name> folder)
    :callback : Function to call when the model is load
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        assert callable(callback), "Callback must be a function"
        callback(enc, sess, context, output, locals())

def read_prompts(hint):
    lines = []
    try:
        line = input(hint)
    except KeyboardInterrupt:
        print()
        line = None
    # Ctrl+D on the first line will end the program (by EOFError exception)
    while True:
        if line is not None:
            lines.append(line)
        # else:
        #     break
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            line = None
    return '\n'.join(lines)

def read_chat_prompt(user_tag="Query", bot_tag="Response", allow_empty=False):
    raw_prompt = input(user_tag + ": ")
    if not raw_prompt and not allow_empty:
        while not raw_prompt:
            print('Prompt should not be empty!')
            raw_prompt = input(user_tag + ": ")
    print(bot_tag + ": ", end='')
    return user_tag + ': ' + raw_prompt + '\n' + bot_tag + ': '

def crop_response_by_newline(response):
    return response.split('\n')[0]

def crop_response_by_eot(tokens):
    eot = 0
    while eot < len(tokens) and tokens[eot] != 50256:
        eot += 1
    return tokens[:eot]

def get_sample_news_feed():
    return "1. It\'s sunny today.\n2. There is a huge meteorite flying by the earth."

def get_news_feed():
    return get_reddit_news(settings['news-feed-agent']['params'])

def news_topic_chatty(enc, sess, context, output, params):
    """
    Chat about today's news
    :fixed_prompt=None : String, a prompt that are used every time.
    Prepended to the prompt input every time.
    :rolling_prompt=0 : Integer, the number of previous conversations to keep.
    The past conversations will be placed between fixed_prompt and the next prompt input.
    :process_response=None : Function for post-processing of the model responses.
    """
    nsamples, batch_size = params['nsamples'], params['batch_size']
    verbose = True
    user_tag = "Query"
    bot_tag = "Response"

    def get_example_conversation():
        return f"{user_tag}: Hello. I am {user_tag}. What's your name?\n{bot_tag}: Hello. My name is {bot_tag}."

    fixed_prompt = "The news today:\n" + get_news_feed() + "\nNow let's talk about it.\n\n" + get_example_conversation() + "\n\n"
    rolling_prompt = 2

    past_memory = []
    while True:
        raw_prompt = read_chat_prompt()
        raw_text = raw_prompt
        if rolling_prompt:
            raw_text = '\n\n'.join(past_memory + [raw_text])
        if fixed_prompt is not None:
            raw_text = fixed_prompt + raw_text

        if verbose:
            print("="*40 + " Model Prompt " + "="*40)
            print(raw_text)
            print("="*80)

        context_tokens = enc.encode(raw_text)
        generated = 0
        preserved_response = None
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                response = crop_response_by_eot(out[i])
                text = enc.decode(response)
                text = crop_response_by_newline(text)
                # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                # print(text)
                print(text)
                if preserved_response is None:
                    preserved_response = text
        # print("=" * 80)

        past_memory.append(raw_prompt + preserved_response)
        if len(past_memory) > rolling_prompt:
            past_memory.pop(0)

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='gpt-2/models',
    fixed_prompt=None,
    rolling_prompt=0,
    process_response=None,
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
     :fixed_prompt=None : String, a prompt that are used every time.
     Prepended to the prompt input every time.
     :rolling_prompt=0 : Integer, the number of previous conversations to keep.
     The past conversations will be placed between fixed_prompt and the next prompt input.
     :process_response=None : Function for post-processing of the model responses.
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        past_memory = []
        while True:
            raw_prompt = read_prompts("Model prompt >>> ")
            while not raw_prompt:
                print('Prompt should not be empty!')
                raw_prompt = read_prompts("Model prompt >>> ")

            raw_text = raw_prompt
            if rolling_prompt:
                raw_text = '\n'.join(past_memory) + raw_text
            if fixed_prompt is not None:
                raw_text = fixed_prompt + raw_text

            context_tokens = enc.encode(raw_text)
            generated = 0
            preserved_response = None
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    if process_response:
                        process_response(text)
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                    if preserved_response is None:
                        preserved_response = text
            print("=" * 80)

            past_memory.append(raw_prompt + preserved_response)
            if len(past_memory) > rolling_prompt:
                past_memory.pop()

if __name__ == "__main__":
    run_model(model_name="124M", callback=news_topic_chatty)
