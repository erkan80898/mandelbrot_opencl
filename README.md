# Rust Mandelbrot Viewer #

An application that displays the Mandelbrot set. The user can zoom into varies parts of the set.

Opencl is used to parallelize the computation, which results in a very fast application.

When you build and run the application, you will need to pass it: 

* The desired resolution width 
* The desired resolution height
* The max number of iteration before we conclude that it does not diverge (A higher value will give better detail, but will lower preformance)

To end the program, enter Ctrl-C
