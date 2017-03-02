import numpy as np
from geoip2.errors import AddressNotFoundError
from ua_parser import user_agent_parser
import re
from urllib.parse import urlparse
import geoip2.database
from datetime import datetime
import time
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import FeatureHasher
from collections import defaultdict, Counter

REGEX_LOG_FORMAT_VARIABLE = r'\$([a-zA-Z0-9\_]+)'
LOG_FORMAT = '$remote_addr - $time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" "$http_x_forwarded_for" $http_host'

REGEX_SPECIAL_CHARS = r'([\.\*\+\?\|\(\)\{\}\[\]])'


def build_pattern(log_format):
    pattern = re.sub(REGEX_SPECIAL_CHARS, r'\\\1', log_format)
    pattern = re.sub(REGEX_LOG_FORMAT_VARIABLE, '(?P<\\1>.*)', pattern)
    return re.compile(pattern)


partent = build_pattern(LOG_FORMAT)

reader_country = geoip2.database.Reader("/home/binh/python/cloak/geo/GeoIP2-Country.mmdb")


def get_country(ip):
    try:
        return reader_country.country(ip).country.iso_code
    except AddressNotFoundError:
        return u'NA'


def load_data():
    with open('access.log', 'r') as file:
        for count, line in enumerate(file):
            line = line.strip()
            if line:
                match = partent.match(line)
                item = match.groupdict()
                if item['http_host'] == 'mydomain.com':
                    item['status'] = int(item['status'])
                    item['body_bytes_sent'] = int(item['body_bytes_sent'])
                    match = re.match('^(\w+)\s+(.*?)\s+(.*?)$', item['request'])
                    if match:
                        item['method'], item['url'], item['version'] = match.groups()
                    item['time_local'] = datetime.strptime(item['time_local'].split()[0], '%d/%b/%Y:%X')
                    item['time_local'] = int(time.mktime(item['time_local'].timetuple()))
                    if item['http_referer'] != '-':
                        item['ref'] = urlparse(item['http_referer']).netloc

                    country = get_country(item['remote_addr'])
                    item["country"] = country if country else 'N/a'
                    parsed_string = user_agent_parser.Parse(item['http_user_agent'])
                    item['browser_family'] = parsed_string['user_agent']['family'] if parsed_string['user_agent'][
                        'family'] else 'N/a'
                    item['os_family'] = parsed_string['os']['family'] if parsed_string['os']['family'] else 'N/a'
                    del item['http_user_agent']
                    del item['http_referer']
                    del item['time_local']
                    del item['request']
                    del item['version']
                    yield item


tic = time.time()
vec = FeatureHasher()
items = list(load_data())
# trains, tests = train_test_split(items, train_size=0.8)
X_train = vec.fit_transform(items)
print("Total", len(items))
# print("Train", len(trains))
# print("Test", len(tests))
print("Done fit train")
# X_test = vec.transform(tests)
print("Done fit test")

# fit the model
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

rng = np.random.RandomState(42)
clf = IsolationForest(random_state=rng, n_estimators=10)
print("Start Fit Model")
clf.fit(X_train)
print("Done Fit Model")
count = 0
counter = defaultdict(int)
for pos, it in enumerate(clf.predict(X_train)):
    if it < 0:
        count += 1
        counter[items[pos]['remote_addr']] += 1
        print(it, items[pos])
print(count)
for ip, count in Counter(counter).most_common():
    print(ip, count)
print("Done All!")
print(time.time() - tic)
