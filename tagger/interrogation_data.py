from math import ceil
import json


def get_i_wt(stored: float):
    """
    in db.json or InterrogationDB.weighed, with weights + increment in the list
    similar for the "query" dict. Same increment per filestamp-interrogation.
    """
    i = ceil(stored) - 1
    return (i, stored - i)


class InterrogationDB:
    def __init__(self, path=None):
        if path:
            data = json.loads(path.read_text())
            if "tag" not in data or "rating" not in data or len(data) < 3:
                raise TypeError
            self.weighed = (data["tag"], data["rating"])
            self.query = data["query"]
        else:
            # lists of index-weights for ratings and tags respectively
            self.weighed = ({}, {})
            # per filestamp-interrogation a unique index (same as in weighed),
            # and a path, if available (not for image queries)
            self.query = {}

    def __contains__(self, key):
        return key in self.query

    def write_json(self, path):
        path.write_text(json.dumps({
            "tag": self.weighed[0],
            "rating": self.weighed[1],
            "query": self.query
        }))

    @property
    def query_count(self):
        return len(self.query)

    def get(self, fi_key: str):
        idx = self.query[fi_key][1]
        ret = ({}, {})

        for i in range(2):
            for ent, lst in self.weighed[i].items():
                x = next((x for i, x in map(get_i_wt, lst) if i == idx), None)
                if x is not None:
                    ret[i][ent] = x
        return ret

    def collect(self, collect_db):
        for index in range(2):
            for ent, lst in self.weighed[index].items():
                for i, val in map(get_i_wt, lst):
                    if i in collect_db:
                        collect_db[i][2+index][ent] = val

    def get_index(self, fi_key: str, path=''):
        if path and path != self.query[fi_key][0]:
            if self.query[fi_key][0] != '':
                print(f'Dup or rename: Identical checksums for {path}\n'
                      'and: {self.query[fi_key][0]} (path updated)')
            self.query[fi_key][0] = path

        # this file was already queried for this interrogator.
        return self.query[fi_key][1]

    def add(self, index: int, ent: str, val: float):
        if ent not in self.weighed[index]:
            self.weighed[index][ent] = []

        self.weighed[index][ent].append(val + len(self.query))

    def story_query(self, fi_key, path):
        self.query[fi_key] = (path, len(self.query))
