"""API module for FastAPI"""
from typing import Callable, Dict, Optional, Tuple
from threading import Lock
from secrets import compare_digest
import asyncio
from collections import defaultdict
from itertools import cycle

from modules import shared  # pylint: disable=import-error
from modules.api.api import decode_base64_to_image  # pylint: disable=E0401
from modules.call_queue import queue_lock  # pylint: disable=import-error
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from tagger import utils  # pylint: disable=import-error
from tagger import api_models as models  # pylint: disable=import-error


class Api:
    """Api class for FastAPI"""
    def __init__(
        self, app: FastAPI, qlock: Lock, prefix: Optional[str] = None
    ) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue: Dict[str, asyncio.Queue] = {}
        self.results = Dict[str, Dict[str, Dict[str, float]]] = {}
        self.queue_lock = qlock
        self.running_batches: Dict[str, Dict[str, asyncio.Task]] = \
            defaultdict(dict)

        self.runner: Optional[asyncio.Task] = None
        self.prefix = prefix

        self.images: Dict[str, Dict[str, Dict[str, tuple[object, float]]]] = \
            defaultdict(lambda: defaultdict(dict))

        self.finished_batches: Dict[str, Dict[str, asyncio.Task]] = \
            defaultdict(dict)

        self.reached_end: Dict[str, Dict[str, bool]] = defaultdict(dict)

        self.add_api_route(
            'interrogate',
            self.endpoint_interrogate,
            methods=['POST'],
            response_model=models.TaggerInterrogateResponse
        )

        self.add_api_route(
            'interrogators',
            self.endpoint_interrogators,
            methods=['GET'],
            response_model=models.TaggerInterrogatorsResponse
        )

        self.add_api_route(
            'unload-interrogators',
            self.endpoint_unload_interrogators,
            methods=['POST'],
            response_model=str,
        )

    async def process_model(self, model: str) -> None:
        """Process a batch of images"""
        res: Dict[str, Dict[str, float]] = defaultdict(dict)
        while len(self.queue[model]) > 0:
            skipped = 0
            for queue in self.queue[model]:
                while True:
                    try:
                        queue_name, name, image, threshold = await queue.get_nowait()
                    except asyncio.QueueEmpty:
                        skipped += 1
                        break # if empty move on to next queue for same model (if any)
                    if name == "":
                        # This is the end of the queue
                        self.results[queue_name] = res[queue_name]
                        del res[queue_name]
                        skipped += 1
                        break
                    # No queue or queued_name, processes instead of queuing
                    res[queue_name][name] = await self.endpoint_interrogate(
                        models.TaggerInterrogateRequest(
                            image=image,
                            model=model,
                            threshold=threshold,
                            queue="",
                            queued_name=""
                        )
                    )
            if skipped == len(self.queue[model]):
                # if all queues for this model are empty, postpone interrogration
                # for this model and do other models first
                break

    async def batch_process(self) -> None:
        while len(self.queue) > 0:
            for model in self.queue:
                if model not in self.running_batches:
                    self.running_batches[model] = asyncio.create_task(
                        self.process_model(model)
                    )
                elif len(self.queue[model]) == 0:
                    await self.running_batches[model]
                    del self.running_batches[model]
            await asyncio.sleep(0.1)


    def auth(self, creds: Optional[HTTPBasicCredentials] = None):
        if creds is None:
            creds = Depends(HTTPBasic())
        if creds.username in self.credentials:
            if compare_digest(creds.password,
                              self.credentials[creds.username]):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'

        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[
                   Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def endpoint_interrogate(self, req: models.TaggerInterrogateRequest):
        """ one file interrogation """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        image = decode_base64_to_image(req.image)
        res: Dict[str, Dict[str, float]] = defaultdict(dict)
        m, q, n = (req.model, req.queue, req.name)

        if n == '' and q in self.running_batches[m]:
            # wait for batch to finish
            res = self.running_batches[m][q].result()
            del self.running_batches[m][q]

        elif q != '':
            # queueing interrogation of the image
            with self.queue_lock:
                # check before populating the queue
                initialize_runner = len(self.queue) == 0

                # create to retreive data when a queue is finished
                if q not in self.queue[m]:
                    self.queue[m][q] = asyncio.Queue()

                # clobber existing image
                if n in self.images[m][q]:
                    i = 0
                    while f'{n}.{i}' in self.images[m][q]:
                        i = i + 1
                    n = f'{n}.{i}'

                # add image to queue
                self.queue[m][q].put_nowait((n, image, req.threshold))
                if initialize_runner:
                    self.runner = asyncio.create_task(self.batch_process())
        else:
            interrogator = utils.interrogators[m]
            with self.queue_lock:
                rating, tag = interrogator.interrogate(image)

            if req.threshold > 0.0:
                tag = {k: v for k, v in tag.items() if v > req.threshold}
            res[n] = {**rating, **tag}

        return models.TaggerInterrogateResponse(caption=res)

    def endpoint_interrogators(self):
        return models.TaggerInterrogatorsResponse(
            models=list(utils.interrogators.keys())
        )

    def endpoint_unload_interrogators(self):
        unloaded_models = 0

        for i in utils.interrogators.values():
            if i.unload():
                unloaded_models = unloaded_models + 1

        return f"Successfully unload {unloaded_models} model(s)"


def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/tagger/v1')
