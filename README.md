# README #

これは線形方程式を各種Krylov部分空間法を用いて解くためのライブラリです．

このライブラリはデフォルトでIntel Math Kernel Library (MKL) に依存しています．
MKLがインストールされている環境であれば，

make

でビルド可能です．ビルドが終わるとlibディレクトリが生成され，
その中に.aファイルが1つ，生成されます．また，mklを使わない場合でも
ビルドすることが可能ですが，この場合，blasとlapackが必要になります．
Makefileを適宜書き換えてblasとlapackのパスを通してください．
mklを使わない場合は

make mkl=no

でビルド可能です．

ビルドにはデフォルトでiccを使用しています．gcc等の他の
コンパイラを使用する場合は，Makefileを書き換えるか上記のコマンドに続けて

CC=gcc

等と書くとそのコンパイラでビルドします．

Mac OSXでも動作します．こちらもデフォルトではmklを使用します．
また，blasについては，デフォルトではXcodeに付属する
blas(Accelerateフレームワーク)を使用します．
lapackについては別途必要になります．
(本ライブラリで使用しているのはlapackeで，Accelerateフレームワーク内のlapackは
clapackであるため)

Accelerateフレームワークを使用しない場合は，
Makefile内の変数OSX_ACCEについて

OSX_ACCE = no

としてください．

make libraryでライブラリのビルドのみ，
make testでテストプログラムのみコンパイルします．
実行ファイルはtestsに生成されます．

Wikiのトップページにあるサンプルプログラムをコンパイルするには，
ヘッダ及び環境変数LD_LIBRARY_PATH(OSXではDYLD_LIBRARY_PATH)に
パスを通した上で

gcc -o sample sample.c -lbksp

等とコマンドを入力します．