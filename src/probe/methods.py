import http.client, xmlrpc.client
class T(xmlrpc.client.Transport):
    def __init__(self, timeout=5.0):
        super().__init__(); self.timeout = timeout
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)
p = xmlrpc.client.ServerProxy("http://192.168.58.3:20003", transport=T())
ms = sorted(p.system.listMethods())
print(f"共 {len(ms)} 个方法\n")
for i in range(0, len(ms), 4):
    print("  " + "".join(f"{m:<28}" for m in ms[i:i+4]))
