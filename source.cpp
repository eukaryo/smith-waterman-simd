#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stack>
#include <queue>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <cassert>
#include <cmath>
#include <iterator>
#include <functional>
#include <complex>
#include <random>
#include <chrono>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <cstring>
#include <cstdint>

#include <mmintrin.h> //MMX
#include <xmmintrin.h>//SSE
#include <emmintrin.h>//SSE2
#include <pmmintrin.h>//SSE3
#include <tmmintrin.h>//SSSE3
#include <smmintrin.h>//SSE4.1
#include <nmmintrin.h>//SSE4.2
#include <wmmintrin.h>//AES
#include <immintrin.h>//AVX, AVX2, FMA
//#include <zmmintrin.h>//AVX-512

int SmithWaterman(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {
	//Smith-Waterman, 全埋めDPをやってスコアだけを返す。トレースバックなし
	//linear gap
	//DNAなのでscore matrixは4*4対称行列で16要素

#define INDEX(ii, jj) ((ii) * 129 + (jj))

	int dp[129 * 129] = {};

	int answer = 0;

	for (int i = 1; i <= 128; ++i) {
		for (int j = 1; j <= 128; ++j) {
			const int index = INDEX(i, j);
			dp[index] = 0;
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 1)] + score_matrix[obs1[i - 1] * 4 + obs2[j - 1]]);
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 0)] - gap_penalty);
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 0, j - 1)] - gap_penalty);
			answer = std::max<int>(answer, dp[index]);
		}
	}

#undef INDEX

	return answer;
}

int SmithWaterman_simd(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {

	//Smith-Waterman, 全埋めDPをやってスコアだけを返す。トレースバックなし
	//linear gap
	//DNAなのでscore matrixは4*4対称行列で16要素
	//16bit/cellでAVX2による16並列
	//score matrixは対角成分が正で非対角成分が負、gap penaltyは正を想定

	/*

	[--][a3][a2][a1][a0][b3][b2][b1][b0]
	[--][--][d2][c3][r3][r7][r*][r*]
	[--][d1][c2][r2][r6][r*][r*]
	[d0][c1][r1][r5][r*][r*]
	[c0][r0][r4][r*][r*]
	計算方法の概略を説明する。簡単のため4ワードのSIMDの場合の図を上に書いた。（実際には16ワードで行う）
	上の図で、ブラケット"[]"は1ワードを意味する。3が最上位ワードで0が最下位ワードとする。
	最内側ループが始まる時点で、変数value_yoko, next_value_yoko, naname1, naname2を以下の通りに定義する。
	value_yoko := [a3,a2,a1,a0]
	next_value_yoko := [b3,b2,b1,b0]
	naname1 := [c3,c2,c1,c0]
	naname2 := [a1,d2,d1,d0]
	最内側ループではresult:=[r3,r2,r1,r0]の4セルの値を一気に計算する。そのためにnaname1_rightshiftedを以下の通りに求める。
	naname1_rightshifted := [a0,c3,c2,c1]
	これはpermuteとalignrで"2変数にまたがるワード単位の論理シフト"をやることで求められる。
	naname1, naname1_rightshifted, naname2の3変数を参照すればresultを求められる。
	その次には[r7,r6,r5,r4]を求めたいわけだが、
	現在のresultを次回のnaname1にして、現在のnaname1_rightshiftedを次回のnaname2にすることができる。
	次回のvalue_yokoとnext_value_yokoは、現在の値を1ワード左シフトすればよい。
	最内側ループに入ってから抜けるまでの間に、rで始まる平行四辺形状の16セルを計算する。そうすることで、
	value_yokoとnext_value_yokoのロードをシンプルにできる。
	*/

	//先頭15文字と末尾17文字をパディングして160文字にする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[160];
	//for (int i = 0; i < 15; ++i)obs2p[i] = 0x80;
	*(uint64_t *)(&obs2p[0]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[8]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 15; i < 143; ++i)obs2p[i] = obs2[i - 15];
	//for (int i = 15; i < 143; i += 8)*(uint64_t *)(&obs2p[i]) = *(uint64_t *)(&obs2[i - 15]);
	std::memcpy(&obs2p[15], &obs2[0], 128);

	//for (int i = 143; i < 160; ++i)obs2p[i] = 0x80;
	obs2p[143] = 0x80;
	*(uint64_t *)(&obs2p[144]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[152]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_16bit = _mm256_setzero_si256();
	const __m256i delta_plus_gap_16bit = _mm256_set1_epi16(127);
	const __m256i gap_16bit = _mm256_set1_epi16(gap_penalty);
	const __m256i delta_16bit = _mm256_sub_epi16(_mm256_set1_epi16(127), gap_16bit);
	const __m256i scorematrix_plus_gap_and_delta_8bit = _mm256_add_epi8(
		_mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&score_matrix[0])),
		_mm256_set_epi64x(0, 0, 0x7f7f'7f7f'7f7f'7f7fLL, 0x7f7f'7f7f'7f7f'7f7fLL));

	__m256i yoko[10];
	for (int i = 0; i < 10; ++i)yoko[i] = _mm256_setzero_si256();

	for (int i = 0; i < 128; i += 16) {

		const __m256i tmp1 = _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&obs1[i]));
		const __m256i sequence_tate = _mm256_shuffle_epi8(tmp1, _mm256_set_epi64x(0, 0, 0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i tate2 = _mm256_slli_epi64(sequence_tate, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m256i value_yoko = yoko[0];

		for (int j = 1; j < 10; ++j) {

			__m256i next_value_yoko = yoko[j];
			__m256i sequence_yoko = _mm256_loadu_si256((__m256i *)&obs2p[(j - 1) * 16]);

			for (int k = 0; k < 16; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。引かれる値はuint8_tで、上位128bitは不定だがあとで潰れるのでよい。
				const __m256i index_score_matrix_8bit = _mm256_add_epi8(tate2, sequence_yoko);
				const __m256i value_score_matrix_plus_gap_and_delta_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_and_delta_8bit, index_score_matrix_8bit);

				//yokoを32文字ロードしていたが、1文字ぶん右シフトしておく。この場所に入れる必然性はないが、software pipeliningを期待して早めに入れる。
				sequence_yoko = _mm256_alignr_epi8(_mm256_permute2x128_si256(sequence_yoko, sequence_yoko, 0b1000'0001), sequence_yoko, 1);

				//スコアマトリックスの値はuint8_tだったがuint16_tに"キャスト"する。上位128bitは不定だったがここで潰れる。
				const __m256i tmp2 = _mm256_permute4x64_epi64(value_score_matrix_plus_gap_and_delta_8bit, 0b0001'0000);
				const __m256i value_score_matrix_plus_gap_and_delta_16bit = _mm256_unpacklo_epi8(tmp2, _mm256_setzero_si256());

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, value_yoko, 0b0010'0001);
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				answer_16bit = _mm256_max_epi16(answer_16bit, result);

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				//value_yokoを1ワード左シフトして、空いた最下位ワードにnext_value_yokoの最上位ワードを入れる
				const __m256i tmp8 = _mm256_permute2x128_si256(value_yoko, next_value_yoko, 0b0000'0011);
				value_yoko = _mm256_alignr_epi8(value_yoko, tmp8, 14);

				//next_value_yokoを1ワード左シフト
				const __m256i tmp9 = _mm256_permute2x128_si256(next_value_yoko, next_value_yoko, 0b0000'1000);
				next_value_yoko = _mm256_alignr_epi8(next_value_yoko, tmp9, 14);

				//resultの最下位ワードをyokoの適切な位置に代入
				//yoko[j - 1].m256i_i16[15 - k] = result.m256i_i16[0];//←これめっっっっっっちゃ遅い

				//resultの最下位ワードをyokoの適切な位置に代入する。
				//具体的には、yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる。あとでワードを逆順にする。
				const __m256i tmpa = _mm256_permute2x128_si256(yoko[j - 1], result, 0b0010'0001);
				yoko[j - 1] = _mm256_alignr_epi8(tmpa, yoko[j - 1], 2);
			}

			//ここでyokoのワードを逆順にする。
			//for (int x = 0; x < 8; ++x)std::swap(yoko[j - 1].m256i_i16[x], yoko[j - 1].m256i_i16[15 - x]);
			const __m256i tmpb = _mm256_permute4x64_epi64(yoko[j - 1], 0b0001'1011);
			const __m256i tmpc = _mm256_shufflehi_epi16(tmpb, 0b0001'1011);
			yoko[j - 1] = _mm256_shufflelo_epi16(tmpc, 0b0001'1011);

			value_yoko = yoko[j];
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd2(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd版からの変更点：
	//(1)引数配列のパディングを末尾16文字ぶん長くして、合計176文字にした。
	//(2)最内側ループでsequence_yoko を1バイトシフトするのではなく、毎回_mm256_loadu_si256で1文字ずつずれた領域をロードするようにした。




	//先頭15文字と末尾33文字をパディングして176文字にする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[176];
	//for (int i = 0; i < 15; ++i)obs2p[i] = 0x80;
	*(uint64_t *)(&obs2p[0]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[8]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 15; i < 143; ++i)obs2p[i] = obs2[i - 15];
	//for (int i = 15; i < 143; i += 8)*(uint64_t *)(&obs2p[i]) = *(uint64_t *)(&obs2[i - 15]);
	std::memcpy(&obs2p[15], &obs2[0], 128);

	//for (int i = 143; i < 176; ++i)obs2p[i] = 0x80;
	obs2p[143] = 0x80;
	*(uint64_t *)(&obs2p[144]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[152]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[160]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[168]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_16bit = _mm256_setzero_si256();
	const __m256i delta_plus_gap_16bit = _mm256_set1_epi16(127);
	const __m256i gap_16bit = _mm256_set1_epi16(gap_penalty);
	const __m256i delta_16bit = _mm256_sub_epi16(_mm256_set1_epi16(127), gap_16bit);
	const __m256i scorematrix_plus_gap_and_delta_8bit = _mm256_add_epi8(
		_mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&score_matrix[0])),
		_mm256_set_epi64x(0, 0, 0x7f7f'7f7f'7f7f'7f7fLL, 0x7f7f'7f7f'7f7f'7f7fLL));

	__m256i yoko[10];
	for (int i = 0; i < 10; ++i)yoko[i] = _mm256_setzero_si256();

	for (int i = 0; i < 128; i += 16) {

		const __m256i tmp1 = _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&obs1[i]));
		const __m256i sequence_tate = _mm256_shuffle_epi8(tmp1, _mm256_set_epi64x(0, 0, 0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i tate2 = _mm256_slli_epi64(sequence_tate, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m256i value_yoko = yoko[0];

		for (int j = 1; j < 10; ++j) {

			__m256i next_value_yoko = yoko[j];
			__m256i sequence_yoko = _mm256_loadu_si256((__m256i *)&obs2p[(j - 1) * 16]);

			for (int k = 0; k < 16; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。引かれる値はuint8_tで、上位128bitは不定だがあとで潰れるのでよい。
				const __m256i index_score_matrix_8bit = _mm256_add_epi8(tate2, sequence_yoko);
				const __m256i value_score_matrix_plus_gap_and_delta_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_and_delta_8bit, index_score_matrix_8bit);

				//yokoを32文字ロードしていたが、1文字ぶん右シフトしておく。この場所に入れる必然性はないが、software pipeliningを期待して早めに入れる。
				//sequence_yoko = _mm256_alignr_epi8(_mm256_permute2x128_si256(sequence_yoko, sequence_yoko, 0b1000'0001), sequence_yoko, 1);

				//↑はperm命令とalignr命令が詰まる可能性があって、loaduで同じことをやったほうが速い↓
				sequence_yoko = _mm256_loadu_si256((__m256i *)&obs2p[(j - 1) * 16 + k + 1]);

				//スコアマトリックスの値はuint8_tだったがuint16_tに"キャスト"する。上位128bitは不定だったがここで潰れる。
				const __m256i tmp2 = _mm256_permute4x64_epi64(value_score_matrix_plus_gap_and_delta_8bit, 0b0001'0000);
				const __m256i value_score_matrix_plus_gap_and_delta_16bit = _mm256_unpacklo_epi8(tmp2, _mm256_setzero_si256());

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, value_yoko, 0b0010'0001);
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				answer_16bit = _mm256_max_epi16(answer_16bit, result);

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				//value_yokoを1ワード左シフトして、空いた最下位ワードにnext_value_yokoの最上位ワードを入れる
				const __m256i tmp8 = _mm256_permute2x128_si256(value_yoko, next_value_yoko, 0b0000'0011);
				value_yoko = _mm256_alignr_epi8(value_yoko, tmp8, 14);

				//next_value_yokoを1ワード左シフト
				const __m256i tmp9 = _mm256_permute2x128_si256(next_value_yoko, next_value_yoko, 0b0000'1000);
				next_value_yoko = _mm256_alignr_epi8(next_value_yoko, tmp9, 14);

				//resultの最下位ワードをyokoの適切な位置に代入
				//yoko[j - 1].m256i_i16[15 - k] = result.m256i_i16[0];//←これめっっっっっっちゃ遅い

				//resultの最下位ワードをyokoの適切な位置に代入する。
				//具体的には、yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる。あとでワードを逆順にする。
				const __m256i tmpa = _mm256_permute2x128_si256(yoko[j - 1], result, 0b0010'0001);
				yoko[j - 1] = _mm256_alignr_epi8(tmpa, yoko[j - 1], 2);
			}

			//ここでyokoのワードを逆順にする。
			//for (int x = 0; x < 8; ++x)std::swap(yoko[j - 1].m256i_i16[x], yoko[j - 1].m256i_i16[15 - x]);
			const __m256i tmpb = _mm256_permute4x64_epi64(yoko[j - 1], 0b0001'1011);
			const __m256i tmpc = _mm256_shufflehi_epi16(tmpb, 0b0001'1011);
			yoko[j - 1] = _mm256_shufflelo_epi16(tmpc, 0b0001'1011);

			value_yoko = yoko[j];
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd3(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd2版からの変更点：
	//(1)yoko変数を128bit変数にした。それにより、バイトシフトの際にperm系命令がいらなくなった。
	//(2)sequence_yokoを128bitだけ読み込むようにした。パディングを160文字に減らした。
	//(3)sequence_yokoを読み込むタイミングを再内側ループの最初にして、次を読み込む命令をなくした。（出るとき無駄になるので）



	//先頭15文字と末尾17文字をパディングして160文字にする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[160];
	//for (int i = 0; i < 15; ++i)obs2p[i] = 0x80;
	*(uint64_t *)(&obs2p[0]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[8]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 15; i < 143; ++i)obs2p[i] = obs2[i - 15];
	//for (int i = 15; i < 143; i += 8)*(uint64_t *)(&obs2p[i]) = *(uint64_t *)(&obs2[i - 15]);
	std::memcpy(&obs2p[15], &obs2[0], 128);

	//for (int i = 143; i < 176; ++i)obs2p[i] = 0x80;
	obs2p[143] = 0x80;
	*(uint64_t *)(&obs2p[144]) = 0x8080'8080'8080'8080ULL;
	*(uint64_t *)(&obs2p[152]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_16bit = _mm256_setzero_si256();
	const __m256i delta_plus_gap_16bit = _mm256_set1_epi16(127);
	const __m256i gap_16bit = _mm256_set1_epi16(gap_penalty);
	const __m256i delta_16bit = _mm256_sub_epi16(_mm256_set1_epi16(127), gap_16bit);
	const __m256i scorematrix_plus_gap_and_delta_8bit = _mm256_add_epi8(
		_mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&score_matrix[0])),
		_mm256_set_epi64x(0, 0, 0x7f7f'7f7f'7f7f'7f7fLL, 0x7f7f'7f7f'7f7f'7f7fLL));

	__m128i yoko[20];
	for (int i = 0; i < 20; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m256i tmp1 = _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)&obs1[i]));
		const __m256i inverse_sequence_tate = _mm256_shuffle_epi8(tmp1, _mm256_set_epi64x(0, 0, 0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_x4 = _mm256_slli_epi64(inverse_sequence_tate, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m128i value_yoko = yoko[1];

		for (int j = 2; j < 20; ++j) {

			__m128i next_value_yoko = yoko[j];

			for (int k = 0; k < 8; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。引かれる値はuint8_tで、上位128bitは不定だがあとで潰れるのでよい。
				const __m256i sequence_yoko = _mm256_zextsi128_si256(_mm_loadu_si128((__m128i *)&obs2p[(j - 2) * 8 + k]));
				const __m256i index_score_matrix_8bit = _mm256_add_epi8(inverse_sequence_tate_x4, sequence_yoko);
				const __m256i value_score_matrix_plus_gap_and_delta_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_and_delta_8bit, index_score_matrix_8bit);

				//スコアマトリックスの値はuint8_tだったがuint16_tに"キャスト"する。上位128bitは不定だったがここで潰れる。
				const __m256i tmp2 = _mm256_permute4x64_epi64(value_score_matrix_plus_gap_and_delta_8bit, 0b0001'0000);
				const __m256i value_score_matrix_plus_gap_and_delta_16bit = _mm256_unpacklo_epi8(tmp2, _mm256_setzero_si256());

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, _mm256_zextsi128_si256(value_yoko), 0b0010'0001);//←ここで_mm256_zextsi128_si256マ？
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				answer_16bit = _mm256_max_epi16(answer_16bit, result);

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				//value_yokoを1ワード左シフトして、空いた最下位ワードにnext_value_yokoの最上位ワードを入れる
				//const __m256i tmp8 = _mm256_permute2x128_si256(value_yoko, next_value_yoko, 0b0000'0011);
				value_yoko = _mm_alignr_epi8(value_yoko, next_value_yoko, 14);

				//next_value_yokoを1ワード左シフト
				//const __m256i tmp9 = _mm256_permute2x128_si256(next_value_yoko, next_value_yoko, 0b0000'1000);
				next_value_yoko = _mm_slli_si128(next_value_yoko, 2);

				//resultの最下位ワードをyokoの適切な位置に代入
				//yoko[j - 1].m256i_i16[15 - k] = result.m256i_i16[0];//←これめっっっっっっちゃ遅い

				//resultの最下位ワードをyokoの適切な位置に代入する。
				//具体的には、yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる。あとでワードを逆順にする。
				//const __m256i tmpa = _mm256_permute2x128_si256(yoko[j - 1], result, 0b0010'0001);
				yoko[j - 2] = _mm_alignr_epi8(_mm256_castsi256_si128(result), yoko[j - 2], 2);
			}

			//ここでyokoのワードを逆順にする。
			//for (int x = 0; x < 4; ++x)std::swap(yoko[j - 2].m128i_i16[x], yoko[j - 2].m128i_i16[7 - x]);
			yoko[j - 2] = _mm_shuffle_epi8(yoko[j - 2], _mm_set_epi64x(0x0100'0302'0504'0706ULL, 0x0908'0b0a'0d0c'0f0eULL));

			//value_yoko = yoko[j];//最内側ループ内のバイトシフトにより自然とこうなるので代入不要
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd4(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd3版からの変更点：
	//(1)文字を全部16bitとして扱うようにした。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for(int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 30; i < 286; i += 2)*(uint16_t *)(&obs2p[i]) = 0x8000 + obs2[i / 2 - 15];
	for (int i = 0; i < 128; i += 16)_mm256_storeu_si256((__m256i *)(&obs2p[30 + i * 2]), _mm256_add_epi64(_mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)&obs2[i])), _mm256_set1_epi16(0x8000)));

	//for (int i = 286; i < 320; ++i)obs2p[i] = 0x80;
	*(uint16_t *)(&obs2p[286]) = 0x8080;
	for (int i = 288; i < 320; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_16bit = _mm256_setzero_si256();
	const __m256i delta_plus_gap_16bit = _mm256_set1_epi16(127);
	const __m256i gap_16bit = _mm256_set1_epi16(gap_penalty);
	const __m256i delta_16bit = _mm256_sub_epi16(_mm256_set1_epi16(127), gap_16bit);
	const __m128i tmp1 = _mm_add_epi8(_mm_loadu_si128((const __m128i *)&score_matrix[0]), _mm_set1_epi8(0x7f));
	const __m256i scorematrix_plus_gap_and_delta_8bit = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp1, 1);

	__m128i yoko[20];
	for (int i = 0; i < 20; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m128i tmp2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)&obs1[i]), _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_16bit_hi0x00 = _mm256_cvtepi8_epi16(tmp2);
		const __m256i inverse_sequence_tate_16bit_hi0x00_x4 = _mm256_slli_epi64(inverse_sequence_tate_16bit_hi0x00, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m128i value_yoko = yoko[1];

		for (int j = 2; j < 20; ++j) {

			__m128i next_value_yoko = yoko[j];

			for (int k = 0; k < 8; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。
				const __m256i sequence_yoko_16bit_hi0x80 = _mm256_loadu_si256((__m256i *)&obs2p[(j - 2) * 16 + k * 2]);
				const __m256i index_score_matrix_16bit_hi0x80 = _mm256_add_epi8(inverse_sequence_tate_16bit_hi0x00_x4, sequence_yoko_16bit_hi0x80);
				const __m256i value_score_matrix_plus_gap_and_delta_16bit = _mm256_shuffle_epi8(scorematrix_plus_gap_and_delta_8bit, index_score_matrix_16bit_hi0x80);

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, _mm256_zextsi128_si256(value_yoko), 0b0010'0001);//←ここで_mm256_zextsi128_si256マ？
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				answer_16bit = _mm256_max_epi16(answer_16bit, result);

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				//value_yokoを1ワード左シフトして、空いた最下位ワードにnext_value_yokoの最上位ワードを入れる
				//const __m256i tmp8 = _mm256_permute2x128_si256(value_yoko, next_value_yoko, 0b0000'0011);
				value_yoko = _mm_alignr_epi8(value_yoko, next_value_yoko, 14);

				//next_value_yokoを1ワード左シフト
				//const __m256i tmp9 = _mm256_permute2x128_si256(next_value_yoko, next_value_yoko, 0b0000'1000);
				next_value_yoko = _mm_slli_si128(next_value_yoko, 2);

				//resultの最下位ワードをyokoの適切な位置に代入
				//yoko[j - 1].m256i_i16[15 - k] = result.m256i_i16[0];//←これめっっっっっっちゃ遅い

				//resultの最下位ワードをyokoの適切な位置に代入する。
				//具体的には、yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる。あとでワードを逆順にする。
				//const __m256i tmpa = _mm256_permute2x128_si256(yoko[j - 1], result, 0b0010'0001);
				yoko[j - 2] = _mm_alignr_epi8(_mm256_castsi256_si128(result), yoko[j - 2], 2);
			}

			//ここでyokoのワードを逆順にする。
			//for (int x = 0; x < 4; ++x)std::swap(yoko[j - 2].m128i_i16[x], yoko[j - 2].m128i_i16[7 - x]);
			yoko[j - 2] = _mm_shuffle_epi8(yoko[j - 2], _mm_set_epi64x(0x0100'0302'0504'0706ULL, 0x0908'0b0a'0d0c'0f0eULL));

			//value_yoko = yoko[j];//最内側ループ内のバイトシフトにより自然とこうなるので代入不要
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

void TestSimdSmithWaterman() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	for (int iteration = 0; iteration < 10000000; iteration++) {
		std::cout << iteration << std::endl;
		std::array<uint8_t, 128>a, b;
		for (int i = 0; i < 128; ++i) {
			a[i] = dna(rnd);
			b[i] = dna(rnd);
		}
		std::array<int8_t, 16>score_matrix = {
			10,-30,-30,-30,
			-30,10,-30,-30,
			-30,-30,10,-30,
			-30,-30,-30,10 };
		uint8_t gap_penalty = 15;

		const int ans1 = SmithWaterman(a, b, score_matrix, gap_penalty);
		const int ans2 = SmithWaterman_simd(a, b, score_matrix, gap_penalty);
		const int ans3 = SmithWaterman_simd2(a, b, score_matrix, gap_penalty);
		const int ans4 = SmithWaterman_simd3(a, b, score_matrix, gap_penalty);
		const int ans5 = SmithWaterman_simd4(a, b, score_matrix, gap_penalty);
		assert(ans1 == ans2);
		assert(ans1 == ans3);
		assert(ans1 == ans4);
		assert(ans1 == ans5);
	}
	return;
}

void SpeedTest() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	std::array<uint8_t, 128>a, b;
	for (int i = 0; i < 128; ++i) {
		a[i] = dna(rnd);
		b[i] = dna(rnd);
	}
	const std::array<int8_t, 16>score_matrix = {
		10,-30,-30,-30,
		-30,10,-30,-30,
		-30,-30,10,-30,
		-30,-30,-30,10 };
	const uint8_t gap_penalty = 15;
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd2(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd2 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd3(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd3 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd4(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd4 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 100000; ++iteration) {
			volatile int score = SmithWaterman(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "normal version: " << elapsed << " ms / 100K" << std::endl;
	}
	return;
}

void InfinitySW() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	std::array<uint8_t, 128>a, b;
	for (int i = 0; i < 128; ++i) {
		a[i] = dna(rnd);
		b[i] = dna(rnd);
	}
	const std::array<int8_t, 16>score_matrix = {
		10,-30,-30,-30,
		-30,10,-30,-30,
		-30,-30,10,-30,
		-30,-30,-30,10 };
	const uint8_t gap_penalty = 15;
	{
		for (;;) {
			volatile int score = SmithWaterman_simd3(a, b, score_matrix, gap_penalty);
		}
	}
	return;
}


int main(void) {

	//TestSimdSmithWaterman();
	//InfinitySW();
	SpeedTest();

	return 0;
}
