
# Chatty

Chatty is a chatbot that is designed to chat actively.

# Setup

Clone [gpt-2](https://github.com/openai/gpt-2) at the root directory of this repository.

```
$ git clone https://github.com/openai/gpt-2
```

And follow the instructions to install it.

After sucessfully install and download the gpt-2 model, install the requirements for Chatty.

```
$ pip install -r requirements.txt
```

# Configuration

1. Copy [settings.yml.template](settings.yml.template) into [settings.yml](settings.yml).

2. Change `model.name` in [settings.yml](settings.yml) to the model you have downloaded.

3. Change `news-feed-agent.params.user-agent` in [settings.yml](settings.yml) to anything appropriate following [Reddit API rules](https://github.com/reddit-archive/reddit/wiki/API).

# Running

Run Chatty using this command:

```
$ python main.py
```

And enjoy chatting with Chatty.

# Sample Results

Some sample results are available in [sample-results](./sample-results). Note that these results may be generated on older versions of Chatty and the settings may not work with the current version.
