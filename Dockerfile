FROM python:3.10
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./LangChain/requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt

COPY ./LangChain/app .
CMD ["chainlit", "run", "app.py", "--port", "7860"]