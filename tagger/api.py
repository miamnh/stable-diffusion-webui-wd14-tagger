"""API module for FastAPI"""
from typing import Callable, Dict, Optional
from threading import Lock
from secrets import compare_digest
import asyncio
from collections import defaultdict
from hashlib import sha256

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
        self.res: Dict[str, Dict[str, Dict[str, float]]] = \
            defaultdict(dict)
        self.queue_lock = qlock

        self.runner: Optional[asyncio.Task] = None
        self.prefix = prefix
        self.running_batches: Dict[str, Dict[str, float]] = \
            defaultdict(lambda: defaultdict(int))

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

    async def add_to_queue(self, m, q, n='', i=None, th=0.0) -> Dict[
        str, Dict[str, float]
    ]:
        with self.queue_lock:
            if m not in self.queue:
                self.queue[m] = asyncio.Queue()
            await self.queue[m].put((q, n, i, th))
        if i is not None:
            if self.runner is None:
                self.runner = asyncio.create_task(self.batch_process())
            # return how many interrogations are done so far per queue
            return self.running_batches
        # wait for the result to become available
        while q in self.running_batches[m]:
            await asyncio.sleep(0.1)
        return self.res.pop(q)

    async def batch_process(self) -> None:
        while len(self.queue) > 0:
            for m in self.queue:
                # if zero the queue might just be pending
                while self.queue[m].qsize() > 0:
                    with self.queue_lock:
                        q, n, i, t = self.queue[m].get_nowait()
                    if n != "":
                        if self.running_batches[m][q] < 0:
                            print(f"Queue {q} is closed")
                            continue
                        self.running_batches[m][q] += 1.0
                        # queue empty to process, not queue
                        self.res[m][n] = await self.endpoint_interrogate(
                            models.TaggerInterrogateRequest(
                                image=i,
                                model=m,
                                threshold=t,
                                queue="",
                                name_in_queue=n
                            )
                        )
                    else:
                        # if there were any queries, mark it finished
                        del self.running_batches[m][q]

            for model in self.running_batches:
                if len(self.running_batches[model]) == 0:
                    with self.queue_lock:
                        del self.queue[model]
            else:
                await asyncio.sleep(0.1)

        self.running_batches.clear()
        self.runner = None

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
        """ one file interrogation, queueing, or batch results """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        m, q, n = (req.model, req.queue, req.name_in_queue)
        if n == '' and q != '':
            # indicate the end of a queue
            tup = (q, n, None, 0.0)
            return asyncio.create_task(self.add_to_queue(m, tup)).result()

        image = decode_base64_to_image(req.image)
        res: Dict[str, Dict[str, float]] = defaultdict(dict)

        if q != '':
            if m not in self.queue:
                self.queue[m] = asyncio.Queue()
            if n == '<sha256>':
                n = sha256(image.tobytes()).hexdigest()
            elif f'{q}#{n}' in self.res[m]:
                # clobber name if it's already in the queue
                i = 0
                while f'{q}#{n}#{i}' in self.res[m]:
                    i += 1
                n = f'{q}#{n}#{i}'
            # add image to queue
            res = asyncio.create_task(
                self.add_to_queue(m, q, n, image, req.threshold)
            ).result()
        else:
            interrogator = utils.interrogators[m]
            res[n], tag = interrogator.interrogate(image)

            for k, v in tag.items():
                if v > req.threshold:
                    res[n][k] = v

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
