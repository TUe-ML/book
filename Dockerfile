#
# Each instruction in this file generates a new layer that gets pushed to your local image cache
#

#
# Lines preceeded by # are regarded as comments and ignored
#

#
# The line below states we will base our new image on the Latest Official Ubuntu 
FROM texlive/texlive

#
# Identify the maintainer of an image
LABEL maintainer="stivendias@gmail.com"

#
# Update the image to the latest packages
RUN apt update

#
# Install packages
RUN apt install python3-pip pdf2svg -y
RUN pip install -U jupyter-book
RUN pip install sphinx-proof
RUN pip install sphinxcontrib-tikz
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install myst_nb
RUN pip install -U scikit-learn
RUN pip install pandas
RUN pip install JSAnimation
