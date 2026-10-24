import site, os
print("User site:", site.getusersitepackages())
if os.path.exists(site.getusersitepackages()):
    print("User packages:", os.listdir(site.getusersitepackages()))

import importlib.metadata
dists = sorted([d.metadata['Name'] for d in importlib.metadata.distributions()])
print("Installed distributions:", dists)
