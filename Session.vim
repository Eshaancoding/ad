let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Coding/ad-py
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +44 ~/.config/nvim/init.lua
badd +20 main.py
badd +4 Session.vim
badd +67 autodiff/linearize.py
badd +14 autodiff/opt.py
badd +1 autodiff/graph/control/__init__.py
badd +62 autodiff/fusion/fuse_within.py
badd +137 autodiff/expr/simplify.py
badd +9442 term://~/Coding/ad-py//68669:/bin/zsh
badd +7 autodiff/fusion/ops.py
badd +8 TODO.md
badd +49 autodiff/__init__.py
badd +63 autodiff/fusion/fuse_across.py
badd +147 autodiff/kernalize/kernalize.py
badd +127 ~/Coding/ad-py/autodiff/node.py
badd +23 autodiff/kernalize/__init__.py
badd +80 autodiff/helper.py
badd +154 autodiff/fusion/helper.py
badd +1 autodiff/device/__init__.py
badd +3 autodiff/alloc/clean.py
badd +97 autodiff/graph/tensor.py
badd +898 term://~/Coding/ad-py//72320:/bin/zsh
badd +10003 term://~/Coding/ad-py//72426:/bin/zsh
badd +29 autodiff/device/opencl/kernels/reduce_elw_fuse.py
badd +5 ~/Coding/ad-py/autodiff/alloc/insert.py
badd +34 autodiff/device/opencl/kernels/unary.py
badd +58 autodiff/alloc/__init__.py
badd +0 term://~/Coding/ad-py//74751:/bin/zsh
badd +60 autodiff/alloc/opt.py
badd +1563 term://~/Coding/ad-py//76926:/bin/zsh
badd +279 term://~/Coding/ad-py//77477:/bin/zsh
argglobal
%argdel
edit TODO.md
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
setlocal foldmethod=manual
setlocal foldexpr=0
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
silent! normal! zE
let &fdl = &fdl
let s:l = 17 - ((16 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 17
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("main.py", ":p")) | buffer main.py | else | edit main.py | endif
if &buftype ==# 'terminal'
  silent file main.py
endif
balt ~/Coding/ad-py/autodiff/node.py
setlocal foldmethod=manual
setlocal foldexpr=0
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
silent! normal! zE
let &fdl = &fdl
let s:l = 5 - ((2 * winheight(0) + 20) / 41)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 5
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("autodiff/linearize.py", ":p")) | buffer autodiff/linearize.py | else | edit autodiff/linearize.py | endif
if &buftype ==# 'terminal'
  silent file autodiff/linearize.py
endif
balt autodiff/__init__.py
setlocal foldmethod=manual
setlocal foldexpr=0
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
silent! normal! zE
let &fdl = &fdl
let s:l = 35 - ((34 * winheight(0) + 42) / 84)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 35
normal! 051|
wincmd w
argglobal
if bufexists(fnamemodify("~/Coding/ad-py/autodiff/alloc/insert.py", ":p")) | buffer ~/Coding/ad-py/autodiff/alloc/insert.py | else | edit ~/Coding/ad-py/autodiff/alloc/insert.py | endif
if &buftype ==# 'terminal'
  silent file ~/Coding/ad-py/autodiff/alloc/insert.py
endif
balt autodiff/alloc/__init__.py
setlocal foldmethod=manual
setlocal foldexpr=0
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
silent! normal! zE
let &fdl = &fdl
let s:l = 5 - ((4 * winheight(0) + 42) / 84)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 5
normal! 045|
wincmd w
4wincmd w
wincmd =
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
