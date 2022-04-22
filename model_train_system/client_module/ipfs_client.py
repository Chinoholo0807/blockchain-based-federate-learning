import ipfshttpclient
import hashlib
from cacheout import Cache


class IPFSClient:

    def __init__(self, setting):
        self.ipfs_api = setting['ipfs_api']
        self.client = ipfshttpclient.connect(self.ipfs_api)
        self.id = setting['id']

    def get_file(self, file_hash):
        resp = self.client.cat(file_hash)
        assert isinstance(resp, bytes)
        return resp

    def add_file(self, bytes_file):
        assert isinstance(bytes_file, bytes)
        file_hash = self.client.add_file(bytes_file)
        assert isinstance(file_hash, str)
        return file_hash

    def close(self) -> None:
        self.client.close()


local_cache = Cache(maxsize=200)


class MockIPFSClient:

    def __init__(self, setting):
        self.id = setting['id']
        pass

    def get_hash(self, bytes_file):
        hash = hashlib.md5()
        hash.update(bytes_file)
        # hash.update(bytes(str(time.time()),encoding="utf-8"))
        hex_str = hash.hexdigest()
        return hex_str

    def add_file(self, bytes_file):
        hex_str = self.get_hash(bytes_file)
        local_cache.add(hex_str, bytes_file)
        return hex_str

    def get_file(self, file_hash):
        # l.debug(f'get bytes with hex_str {hex_str}')
        bytes_file = local_cache.get(file_hash)
        assert not bytes_file is None
        return bytes_file


if __name__ == "__main__":
    wrapper = IPFSClient({'ipfs_api': '/ip4/127.0.0.1/tcp/5001'})
    data = b'just for test data'
    hash_code = wrapper.add_file(data)
    print('---------------', hash_code, type(hash_code))
    resp = wrapper.get_file(hash_code)
    print('---------------', resp, type(resp))
