FROM getkeops/keops-full

RUN pip install git+https://github.com/cornellius-gp/gpytorch.git@altproj
RUN pip install pykeops==2.1.2
RUN pip install pandas

WORKDIR /workspace
