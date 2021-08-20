# Tephra2_PY
A python-reimplemented model of Tephra2, an advection-diffusion model, named Tephra2_PY.
The input data such as ejecta, eruption start time, plume height(h_p) of each eruption, topography, and wind data has been replaced by dummy data of the same shape, so you will have to get it from elsewhere and substitute it.
You can get the topography data around Sakurajima and other Japanese volcanoes from the following GSI website (Japanese):
    https://fgd.gsi.go.jp/download/menu.php
You can get the wind data around Sakurajima and other Japanese volcanoes from the following JMA website (Japanese):
    http://www.data.jma.go.jp/obd/stats/etrn/index.php?prec_no=88&block_no=47827&year=&month=&day=&view=
You can get some data related to Sakurajima eruption from the following JMA website (Japanese):
    http://www.jma-net.go.jp/kagoshima/vol/kazan_top.html
If you have trouble calculating using this code or you have a question related to the code, please e-mail to the following address:
    takishita.kosei.85s@st.kyoto-u.ac.jp
    
移流拡散モデルTephra2をパイソンで再実装したモデルTephra2_PYのソースコードです。
各噴火の噴出量，噴火開始時刻，噴煙高度や，地形データ，風のデータなどの入力値データは同じ形状のダミーデータに置き換えられていますので，他の場所から取得して代入してください。
桜島や他の日本の火山周辺の地形データは以下の国土地理院のサイトから取得できます：
    https://fgd.gsi.go.jp/download/menu.php
桜島や他の日本の火山周辺の風のデータは以下の気象庁のサイトから取得できます：
    http://www.data.jma.go.jp/obd/stats/etrn/index.php?prec_no=88&block_no=47827&year=&month=&day=&view=
桜島の噴火に関する情報は以下の鹿児島地方気象台のサイトから取得できます：
    http://www.jma-net.go.jp/kagoshima/vol/kazan_top.html
もしこのコードを用いて計算する際に問題が生じた場合や，このコードに関連した質問がある場合は，以下のメールアドレスにメールしてください：
    takishita.kosei.85s@st.kyoto-u.ac.jp
