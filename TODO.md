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

* Fix the backward issue for multihead attention

* Tetris optimization

* use program id, not node id (this is the problem with the OpenCL right now)

============= then test neural net ============= 
============ then test multihead att ===========

* better opt node hehe
    * constant simpl

====== Then test with more tests (use pytest) ====== 
* add more control
* control, if, etc. etc. etc.
* Then do device feeder
    * see if you can train a LLM faster than normal
