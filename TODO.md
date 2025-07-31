# TODO

* ~~better fusion operaton~~
    * ~~This was at sequental forward + backend + opt~~

* ~~fix memory optimization (plan for this as well)~~
    * ~~mem opt~~ 
    * ~~After fusion~~

* ~~set up -= and += (somehow repr that)~~
    * ~~setting as the result id might resultant in a lot of breaking~~
    
* ~~dep list (then dep opt)~~
    * ~~before fusion ideally (less nodes to fuse)~~

* ~~repeat opt!  ~~
    * ~~before fusion ideally (less nodes to fuse)~~

* ~~Constant simplification~~

* ~~Tetris optimization~~

* ~~use program id, not node id (this is the problem with the OpenCL right now)~~

============ then test nn ===========

* Fix the backward issue for multihead attention
    * check main.py currently

* ~~make repeat_opt, mem_opt faster~~
    * ~~in general, this entire process is kind of slow...~~
    * ~~not sure if you can speed up the entire thing~~
        * ~~(ideally pawning off to rust is a good idea)~~

============ then test multihead att ===========

* maybe test if it actually like fucking works and numerical test it heheheheh

====== Then test with more tests (use pytest) ====== 
* add more control
* control, if, etc. etc. etc.
* Then do device feeder
    * see if you can train a LLM faster than normal
