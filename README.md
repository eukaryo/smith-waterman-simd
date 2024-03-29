# smith-waterman-simd
平行四辺形

EPYC 7501(max:2.0GHz), icc 19.0.3.199, -std=c++1y -O3 でsimd4が4.4秒/1Mくらいで、つまり呼び出し1回あたり8800サイクルくらいだった。

Visual Studio 2017のパフォーマンスプロファイリングによると実行時間の90 %くらいが最内側ループのものだった。

最内側ループは8 * 18 * 8 = 1152回通るので、つまりループ1回あたり7サイクルくらい。

simd7はEPYC 7501の上ではsimd4より約1%速いが、Core i7 4770 (Haswell)の上ではsimd4のほうがsimd7より速い。

simd9はsimd7から演算を増やした代わりにクリティカルパスを短くした。Xeon Gold 6136の上ではsimd7よりsimd9のほうが速かったが、EPYC 7501の上ではむしろ遅くなった。EPYC 7501はZen1世代なのでAVX2が内部的に128bitごとに処理されているためだと考えられる。

FYI:
Farrar, M. (2007). Striped Smith-Waterman speeds database searches six times over other SIMD implementations. Bioinformatics (Oxford, England), 23(2), 156–161. https://doi.org/10.1093/bioinformatics/btl582
