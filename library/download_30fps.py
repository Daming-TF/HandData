import os

base_url = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/InterHand2.6M/InterHand2.6M.images.30.fps.v1.0/'
for part1 in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'):
    for part2 in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'):
        if part1 == 'h' and part2 == 'q':
            break
        os.system('wget ' + base_url + 'InterHand2.6M.images.30.fps.v1.0.tar.part' + part1 + part2)

os.system('wget ' + base_url + 'InterHand2.6M.images.30.fps.v1.0.tar.CHECKSUM')
os.system('wget ' + base_url + 'InterHand2.6M.images.30.fps.v1.0/unzip.sh')
os.system('wget ' + base_url + 'verify_download.py')