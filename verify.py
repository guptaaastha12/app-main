import sys
import importlib #for dynamic import 


def get_package_version(package_name):
    try:
        module = importlib.import_module(package_name)
        for attr in ['__version__','version','VERSION']:
            if hasattr(module,attr):
                return getattr(module,attr)
            
    except ModuleNotFoundError:
        return f'{package_name} : Module Not Installed'
    except Exception as e:
        return f'Error {e}'

def check_version_from_requirements(file_path):
    print('======== Package Version:============')
    with open(file_path,'r') as file:
       for line in file:
            package_name = line.strip()
            if not package_name or package_name.startswith('#'):
                continue
            version = get_package_version(package_name)
            print(f'{package_name} : {version}')

def main():
    check_version_from_requirements('requirements.txt')


sys.exit(main())
