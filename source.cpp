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
			//std::cout << dp[index] << " ";
		}
		//std::cout << std::endl;
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
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

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

int SmithWaterman_simd5(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd4版からの変更点：
	//(1)yokoをshort型配列にした。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

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

	//__m128i yoko[20];
	alignas(32)short yoko[176] = {};
	//for (int i = 0; i < 20; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m128i tmp2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)&obs1[i]), _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_16bit_hi0x00 = _mm256_cvtepi8_epi16(tmp2);
		const __m256i inverse_sequence_tate_16bit_hi0x00_x4 = _mm256_slli_epi64(inverse_sequence_tate_16bit_hi0x00, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		for (int j = 2; j < 20; ++j) {

			__m128i value_yoko = _mm_loadu_si128((__m128i *)&yoko[j * 8 - 1 + 1]);

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

				//value_yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる
				value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), value_yoko, 2);
			}

			_mm_storeu_si128((__m128i *)&yoko[j * 8 - 16 + 1], value_yoko);
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd6(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd4版からの変更点：
	//(1)yokoのXMMレジスタ内のワードの並び順を逆にした。next_value_yokoを廃して、value_yokoにalignrでresultを入れるようにした。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

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

		for (int j = 2; j < 20; ++j) {

			__m128i value_yoko = _mm_alignr_epi8(yoko[j], yoko[j - 1], 14);

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

				//value_yokoを1ワード右シフトして、空いた最上位ワードにresultの最下位ワードを入れる
				value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), value_yoko, 2);
			}

			yoko[j - 2] = value_yoko;
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd7(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd6版からの変更点：
	//(1)next_value_yokoを再導入して、value_yokoに関する依存関係を減らした。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

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

	__m128i yoko[21];
	for (int i = 0; i < 21; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m128i tmp2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)&obs1[i]), _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_16bit_hi0x00 = _mm256_cvtepi8_epi16(tmp2);
		const __m256i inverse_sequence_tate_16bit_hi0x00_x4 = _mm256_slli_epi64(inverse_sequence_tate_16bit_hi0x00, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m128i value_yoko = _mm_alignr_epi8(yoko[2], yoko[1], 14);

		for (int j = 2; j < 20; ++j) {

			__m128i next_value_yoko = _mm_alignr_epi8(yoko[j + 1], yoko[j], 14);

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

				value_yoko = _mm_alignr_epi8(next_value_yoko, value_yoko, 2);
				next_value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), next_value_yoko, 2);
			}

			yoko[j - 2] = next_value_yoko;
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd8(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd7版からの変更点：
	//(1)DP本体におけるmaxの順番を変えた。演算が1個増えるが、クリティカルパスが1サイクル短くなる。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

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

	__m128i yoko[21];
	for (int i = 0; i < 21; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m128i tmp2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)&obs1[i]), _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_16bit_hi0x00 = _mm256_cvtepi8_epi16(tmp2);
		const __m256i inverse_sequence_tate_16bit_hi0x00_x4 = _mm256_slli_epi64(inverse_sequence_tate_16bit_hi0x00, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = _mm256_setzero_si256();
		__m256i naname2 = _mm256_setzero_si256();

		__m128i value_yoko = _mm_alignr_epi8(yoko[2], yoko[1], 14);

		for (int j = 2; j < 20; ++j) {

			__m128i next_value_yoko = _mm_alignr_epi8(yoko[j + 1], yoko[j], 14);

			for (int k = 0; k < 8; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。
				const __m256i sequence_yoko_16bit_hi0x80 = _mm256_loadu_si256((__m256i *)&obs2p[(j - 2) * 16 + k * 2]);
				const __m256i index_score_matrix_16bit_hi0x80 = _mm256_add_epi8(inverse_sequence_tate_16bit_hi0x00_x4, sequence_yoko_16bit_hi0x80);
				const __m256i value_score_matrix_plus_gap_and_delta_16bit = _mm256_shuffle_epi8(scorematrix_plus_gap_and_delta_8bit, index_score_matrix_16bit_hi0x80);

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, _mm256_zextsi128_si256(value_yoko), 0b0010'0001);//←ここで_mm256_zextsi128_si256マ？
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				//const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				//const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				//const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				//const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				//const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				const __m256i tmp4x = _mm256_add_epi16(naname1, delta_16bit);
				const __m256i tmp5x = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				const __m256i tmp6x = _mm256_add_epi16(naname1_rightshifted, delta_16bit);
				const __m256i tmp7x = _mm256_max_epi16(tmp4x, tmp5x);
				const __m256i tmp8x = _mm256_max_epi16(tmp6x, tmp7x);
				const __m256i result = _mm256_subs_epu16(tmp8x, delta_plus_gap_16bit);

				answer_16bit = _mm256_max_epi16(answer_16bit, result);

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				value_yoko = _mm_alignr_epi8(next_value_yoko, value_yoko, 2);
				next_value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), next_value_yoko, 2);
			}

			yoko[j - 2] = next_value_yoko;
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_simd9(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2,
	const std::array<int8_t, 16>&score_matrix,
	const int8_t gap_penalty) {


	//上のsimd7版からの変更点：
	//(1)DP本体を縦方向に関するoffset DPにして、クリティカルパスを更に短くした。



	//先頭30文字と末尾34文字をパディングして、かつ各文字を16bitにして、トータルで320バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[320];
	//for (int i = 0; i < 32; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 32; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 30; i < 286; i += 2)*(uint16_t *)(&obs2p[i]) = 0x8000 + obs2[i / 2 - 15];
	for (int i = 0; i < 128; i += 16)_mm256_storeu_si256((__m256i *)(&obs2p[30 + i * 2]), _mm256_add_epi64(_mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)&obs2[i])), _mm256_set1_epi16(0x8000)));

	//for (int i = 286; i < 320; ++i)obs2p[i] = 0x80;
	*(uint16_t *)(&obs2p[286]) = 0x8080;
	for (int i = 288; i < 320; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_16bit = _mm256_setzero_si256();
	const __m256i gap_16bit = _mm256_set1_epi16(gap_penalty);
	const int score_offset = 100;
	const __m128i tmp1 = _mm_add_epi8(_mm_loadu_si128((const __m128i *)&score_matrix[0]), _mm_set1_epi8(gap_penalty + score_offset));
	const __m256i scorematrix_plus_gap_and_scoreoffset_8bit = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp1, 1);

	const __m256i offseted_zero = _mm256_set_epi16(
		gap_penalty * 1, gap_penalty * 2, gap_penalty * 3, gap_penalty * 4,
		gap_penalty * 5, gap_penalty * 6, gap_penalty * 7, gap_penalty * 8,
		gap_penalty * 9, gap_penalty * 10, gap_penalty * 11, gap_penalty * 12,
		gap_penalty * 13, gap_penalty * 14, gap_penalty * 15, gap_penalty * 16);
	const __m256i offseted_zero2 = _mm256_set_epi16(
		0, gap_penalty * 1, gap_penalty * 2, gap_penalty * 3,
		gap_penalty * 4, gap_penalty * 5, gap_penalty * 6, gap_penalty * 7,
		gap_penalty * 8, gap_penalty * 9, gap_penalty * 10, gap_penalty * 11,
		gap_penalty * 12, gap_penalty * 13, gap_penalty * 14, gap_penalty * 15);
	const __m128i gapx16 = _mm_set1_epi16(gap_penalty * 16);

	__m128i yoko[21];
	for (int i = 0; i < 21; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 16) {

		const __m128i tmp2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)&obs1[i]), _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL));//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_16bit_hi0x00 = _mm256_cvtepi8_epi16(tmp2);
		const __m256i inverse_sequence_tate_16bit_hi0x00_x4 = _mm256_slli_epi64(inverse_sequence_tate_16bit_hi0x00, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = offseted_zero;
		__m256i naname2 = offseted_zero2;

		__m128i value_yoko = _mm_alignr_epi8(yoko[2], yoko[1], 14);

		for (int j = 2; j < 20; ++j) {

			__m128i next_value_yoko = _mm_alignr_epi8(yoko[j + 1], yoko[j], 14);

			for (int k = 0; k < 8; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。
				const __m256i sequence_yoko_16bit_hi0x80 = _mm256_loadu_si256((__m256i *)&obs2p[(j - 2) * 16 + k * 2]);
				const __m256i index_score_matrix_16bit_hi0x80 = _mm256_add_epi8(inverse_sequence_tate_16bit_hi0x00_x4, sequence_yoko_16bit_hi0x80);
				const __m256i tmpx1 = _mm256_shuffle_epi8(scorematrix_plus_gap_and_scoreoffset_8bit, index_score_matrix_16bit_hi0x80);
				const __m256i value_score_matrix_plus_gap_16bit = _mm256_sub_epi16(tmpx1, _mm256_set1_epi16(score_offset));

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, _mm256_zextsi128_si256(value_yoko), 0b0010'0001);//←ここで_mm256_zextsi128_si256マ？
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 2);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				//const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				//const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				//const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				//const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				//const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				//const __m256i tmp4x = _mm256_add_epi16(naname1, delta_16bit);
				//const __m256i tmp5x = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				//const __m256i tmp6x = _mm256_add_epi16(naname1_rightshifted, delta_16bit);
				//const __m256i tmp7x = _mm256_max_epi16(tmp4x, tmp5x);
				//const __m256i tmp8x = _mm256_max_epi16(tmp6x, tmp7x);
				//const __m256i result = _mm256_subs_epu16(tmp8x, delta_plus_gap_16bit);

				const __m256i tmp4y = _mm256_sub_epi16(naname1, gap_16bit);
				const __m256i tmp5y = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_16bit);
				const __m256i tmp6y = _mm256_max_epi16(offseted_zero, tmp4y);
				const __m256i tmp7y = _mm256_max_epi16(tmp5y, tmp6y);
				const __m256i result = _mm256_max_epi16(tmp7y, naname1_rightshifted);


				answer_16bit = _mm256_max_epi16(answer_16bit, _mm256_sub_epi16(result, offseted_zero));

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				value_yoko = _mm_alignr_epi8(next_value_yoko, value_yoko, 2);
				next_value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), next_value_yoko, 2);
			}

			yoko[j - 2] = _mm_sub_epi16(next_value_yoko, gapx16);
		}
	}

	alignas(32)short candidates[16] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_16bit);

	int result = 0;
	for (int i = 0; i < 16; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_111(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2) {
	//Smith-Waterman, 全埋めDPをやってスコアだけを返す。トレースバックなし
	//linear gap

	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;

#define INDEX(ii, jj) ((ii) * 129 + (jj))

	int dp[129 * 129] = {};

	int answer = 0;

	for (int i = 1; i <= 128; ++i) {
		for (int j = 1; j <= 128; ++j) {
			const int index = INDEX(i, j);
			dp[index] = 0;
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 1)] + (obs1[i - 1] == obs2[j - 1] ? MATCH : -MISMATCH));
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 0)] - GAP);
			dp[index] = std::max<int>(dp[index], dp[INDEX(i - 0, j - 1)] - GAP);
			answer = std::max<int>(answer, dp[index]);
			//std::cout << dp[index] << " ";
		}
		//std::cout << std::endl;
	}

#undef INDEX

	return answer;
}

int SmithWaterman_8bit111simd(
	const std::array<uint8_t, 128>&obs1,
	const std::array<uint8_t, 128>&obs2) {

	//上のsimd9版からの変更点：
	//(1)DP変数を符号なし8bit型にした。その代わりに(match, mismatch, gap)=(1,1,1)に固定した。

	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;

	//先頭31文字と末尾33文字をパディングして、トータルで192バイトにする。
	//0x80で埋める理由は、スコアマトリックス16要素の表引きをpshufbで行うときに、
	//パディングした部分のインデックスの最上位ビットが立立っているとpshufbの仕様により0が与えられるのを利用するためである。
	alignas(32)uint8_t obs2p[192];
	//for (int i = 0; i < 31; ++i)obs2p[i] = 0x80;
	for (int i = 0; i < 31; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	//for (int i = 31; i < 159; ++i)obs2p[i] = obs2[i - 31];
	std::memcpy(&obs2p[31], &obs2[0], 128);

	//for (int i = 159; i < 192; ++i)obs2p[i] = 0x80;
	obs2p[159] = 0x80;
	for (int i = 160; i < 192; i += 8)*(uint64_t *)(&obs2p[i]) = 0x8080'8080'8080'8080ULL;

	__m256i answer_8bit = _mm256_setzero_si256();
	const __m256i gap_8bit = _mm256_set1_epi8(GAP);
	const __m256i scorematrix_plus_gap_8bit = _mm256_set_epi8(
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH,
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH);//MISMATCH <= GAPだからこれでよい。逆に言えば、MISMATCH > GAPにするときはオフセットを与える必要がたぶんあるだろう。
	const __m256i offseted_zero = _mm256_set_epi8(
		GAP * 0x01, GAP * 0x02, GAP * 0x03, GAP * 0x04, GAP * 0x05, GAP * 0x06, GAP * 0x07, GAP * 0x08,
		GAP * 0x09, GAP * 0x0A, GAP * 0x0B, GAP * 0x0C, GAP * 0x0D, GAP * 0x0E, GAP * 0x0F, GAP * 0x10,
		GAP * 0x11, GAP * 0x12, GAP * 0x13, GAP * 0x14, GAP * 0x15, GAP * 0x16, GAP * 0x17, GAP * 0x18,
		GAP * 0x19, GAP * 0x1A, GAP * 0x1B, GAP * 0x1C, GAP * 0x1D, GAP * 0x1E, GAP * 0x1F, GAP * 0x20);
	const __m256i offseted_zero2 = _mm256_set_epi8(
		GAP * 0x00, GAP * 0x01, GAP * 0x02, GAP * 0x03, GAP * 0x04, GAP * 0x05, GAP * 0x06, GAP * 0x07,
		GAP * 0x08, GAP * 0x09, GAP * 0x0A, GAP * 0x0B, GAP * 0x0C, GAP * 0x0D, GAP * 0x0E, GAP * 0x0F,
		GAP * 0x10, GAP * 0x11, GAP * 0x12, GAP * 0x13, GAP * 0x14, GAP * 0x15, GAP * 0x16, GAP * 0x17,
		GAP * 0x18, GAP * 0x19, GAP * 0x1A, GAP * 0x1B, GAP * 0x1C, GAP * 0x1D, GAP * 0x1E, GAP * 0x1F);
	const __m128i gapx32 = _mm_set1_epi8(GAP * 32);
	const __m256i reverser = _mm256_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL, 0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL);

	__m128i yoko[13];
	for (int i = 0; i < 13; ++i)yoko[i] = _mm_setzero_si128();

	for (int i = 0; i < 128; i += 32) {

		const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&obs1[i]);
		const __m256i tmp2 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(tmp1, tmp1, 0b0000'0001), reverser);//シーケンスを逆順にしておく
		const __m256i inverse_sequence_tate_8bit_x4 = _mm256_slli_epi64(tmp2, 2);//2ビット左シフト(=4倍)

		__m256i naname1 = offseted_zero;
		__m256i naname2 = offseted_zero2;

		__m128i value_yoko = _mm_alignr_epi8(yoko[2], yoko[1], 15);

		for (int j = 2; j <= 11; ++j) {

			__m128i next_value_yoko = _mm_alignr_epi8(yoko[j + 1], yoko[j], 15);

			for (int k = 0; k < 16; ++k) {

				//スコアマトリックスのテーブル引きを、pshufbを使って16セルぶん一気に行う。
				const __m256i sequence_yoko_8bit = _mm256_loadu_si256((__m256i *)&obs2p[(j - 2) * 16 + k]);
				const __m256i index_score_matrix_8bit = _mm256_add_epi8(inverse_sequence_tate_8bit_x4, sequence_yoko_8bit);
				const __m256i value_score_matrix_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix_8bit);

				//naname1を1ワード右シフトして、空いた最上位ワードにvalue_yokoの最下位ワードを入れる。
				const __m256i tmp3 = _mm256_permute2x128_si256(naname1, _mm256_zextsi128_si256(value_yoko), 0b0010'0001);//←ここで_mm256_zextsi128_si256マ？
				const __m256i naname1_rightshifted = _mm256_alignr_epi8(tmp3, naname1, 1);

				//nanama1,naname1_rightshifted,naname2などを使いDP値を計算して、resultとする
				//const __m256i tmp4 = _mm256_max_epi16(naname1, naname1_rightshifted);
				//const __m256i tmp5 = _mm256_add_epi16(delta_16bit, tmp4);
				//const __m256i tmp6 = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				//const __m256i tmp7 = _mm256_max_epi16(tmp5, tmp6);
				//const __m256i result = _mm256_subs_epu16(tmp7, delta_plus_gap_16bit);

				//const __m256i tmp4x = _mm256_add_epi16(naname1, delta_16bit);
				//const __m256i tmp5x = _mm256_add_epi16(naname2, value_score_matrix_plus_gap_and_delta_16bit);
				//const __m256i tmp6x = _mm256_add_epi16(naname1_rightshifted, delta_16bit);
				//const __m256i tmp7x = _mm256_max_epi16(tmp4x, tmp5x);
				//const __m256i tmp8x = _mm256_max_epi16(tmp6x, tmp7x);
				//const __m256i result = _mm256_subs_epu16(tmp8x, delta_plus_gap_16bit);

				const __m256i tmp4y = _mm256_subs_epu8(naname1, gap_8bit);
				const __m256i tmp5y = _mm256_adds_epu8(naname2, value_score_matrix_plus_gap_8bit);
				const __m256i tmp6y = _mm256_max_epu8(offseted_zero, tmp4y);
				const __m256i tmp7y = _mm256_max_epu8(tmp5y, tmp6y);
				const __m256i result = _mm256_max_epu8(tmp7y, naname1_rightshifted);


				answer_8bit = _mm256_max_epu8(answer_8bit, _mm256_subs_epu8(result, offseted_zero));

				//naname2 <- naname1_rightshifted
				naname2 = naname1_rightshifted;

				//naname1 <- result
				naname1 = result;

				value_yoko = _mm_alignr_epi8(next_value_yoko, value_yoko, 1);
				next_value_yoko = _mm_alignr_epi8(_mm256_castsi256_si128(result), next_value_yoko, 1);
			}

			yoko[j - 2] = _mm_subs_epu8(next_value_yoko, gapx32);
		}
	}

	alignas(32)uint8_t candidates[32] = {};
	_mm256_storeu_si256((__m256i *)candidates, answer_8bit);

	int result = 0;
	for (int i = 0; i < 32; ++i)result = std::max<int>(result, int(candidates[i]));

	return result;
}

int SmithWaterman_8b111x32mark1(
	const std::array<uint8_t, 128 * 32>&obs1,
	const std::array<uint8_t, 128>&obs2,
	std::array<int, 32>& dest) {

	//上のSmithWaterman_8bit111simdからの変更点:
	//(1)obs1として32本の128merをまとめて受け取り、結果をまとめてdestに入れて返すようにした。
	//   （ちなみに返り値はdest[0]だがこれはなんとなくで深い意味はない）
	//   平行四辺形埋めだと実質160merぶんの計算をやっており、はみ出た領域に対する計算が20%を占めていた。
	//   32本の入力を転置してスライシングすればはみ出ないし、斜め埋めコンセプトのためのpermuteなどが不要になり、効率化すると思われた。

	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;

	alignas(32)uint8_t transposed_obs1[32 * 128];
	for (int i = 0; i < 128; ++i)for (int j = 0; j < 32; ++j)transposed_obs1[i * 32 + j] = obs1[j * 128 + i];

	__m256i answer_8bit = _mm256_setzero_si256();
	const __m256i gap_8bit = _mm256_set1_epi8(GAP);
	const __m256i scorematrix_plus_gap_8bit = _mm256_set_epi8(
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH,
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH);//MISMATCH <= GAPだからこれでよい。

	__m256i yoko[130];
	for (int i = 0; i < 130; ++i)yoko[i] = _mm256_setzero_si256();


	for (int i = 0; i < 128; ++i) {
		__m256i prev = _mm256_setzero_si256();
		__m256i prev2 = _mm256_setzero_si256();
		const __m256i sequence_tate_x4 = _mm256_slli_epi64(_mm256_loadu_si256((__m256i *)&transposed_obs1[i * 32]), 2);

		__m256i naname = yoko[1];
		__m256i ue;

		for (int j = 2; j < 130; ++j) {

			ue = yoko[j];

			const __m256i sequence_yoko = _mm256_set1_epi8(obs2[j - 2]);
			const __m256i index_score_matrix_8bit = _mm256_add_epi8(sequence_tate_x4, sequence_yoko);
			const __m256i value_score_matrix_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix_8bit);

			const __m256i tmp4y = _mm256_max_epu8(prev, ue);
			const __m256i tmp5y = _mm256_adds_epu8(naname, value_score_matrix_plus_gap_8bit);
			const __m256i tmp6y = _mm256_max_epu8(tmp5y, tmp4y);
			const __m256i result = _mm256_subs_epu8(tmp6y, gap_8bit);

			answer_8bit = _mm256_max_epu8(answer_8bit, result);

			yoko[j - 2] = prev2;
			prev2 = prev;
			prev = result;
			naname = ue;

		}
		yoko[128] = prev2;
		yoko[129] = prev;
	}

	alignas(32)uint8_t answers[32] = {};
	_mm256_storeu_si256((__m256i *)answers, answer_8bit);

	for (int i = 0; i < 32; ++i)dest[i] = answers[i];
	return answers[0];
}

int SmithWaterman_8b111x32mark2(
	const std::array<uint8_t, 128 * 32>&obs1,
	const std::array<uint8_t, 128>&obs2,
	std::array<int, 32>& dest) {

	//上のmark1からの変更点：
	//(1)DP本体を縦方向に2個ずつアンロールした。
	//   レジスタリネーミングとアウトオブオーダー実行があれば、クリティカルパスの影響が減ると期待できる。


	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;

	alignas(32)uint8_t transposed_obs1[32 * 128];
	for (int i = 0; i < 128; ++i)for (int j = 0; j < 32; ++j)transposed_obs1[i * 32 + j] = obs1[j * 128 + i];

	__m256i answer_8bit = _mm256_setzero_si256();
	const __m256i gap_8bit = _mm256_set1_epi8(GAP);
	const __m256i scorematrix_plus_gap_8bit = _mm256_set_epi8(
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH,
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH);//MISMATCH <= GAPだからこれでよい。

	__m256i yoko[130];
	for (int i = 0; i < 130; ++i)yoko[i] = _mm256_setzero_si256();


	for (int i = 0; i < 128; i += 2) {
		__m256i prev0 = _mm256_setzero_si256();
		__m256i prev02 = _mm256_setzero_si256();
		__m256i prev1 = _mm256_setzero_si256();
		__m256i prev12 = _mm256_setzero_si256();
		const __m256i sequence_tate0_x4 = _mm256_slli_epi64(_mm256_loadu_si256((__m256i *)&transposed_obs1[(i + 0) * 32]), 2);
		const __m256i sequence_tate1_x4 = _mm256_slli_epi64(_mm256_loadu_si256((__m256i *)&transposed_obs1[(i + 1) * 32]), 2);

		__m256i naname = yoko[1];
		__m256i ue;

		for (int j = 2; j < 130; ++j) {

			ue = yoko[j];

			const __m256i sequence_yoko = _mm256_set1_epi8(obs2[j - 2]);

			const __m256i index_score_matrix0_8bit = _mm256_add_epi8(sequence_tate0_x4, sequence_yoko);
			const __m256i value_score_matrix0_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix0_8bit);
			const __m256i tmp4y0 = _mm256_max_epu8(prev0, ue);
			const __m256i tmp5y0 = _mm256_adds_epu8(naname, value_score_matrix0_plus_gap_8bit);
			const __m256i tmp6y0 = _mm256_max_epu8(tmp5y0, tmp4y0);
			const __m256i result0 = _mm256_subs_epu8(tmp6y0, gap_8bit);
			answer_8bit = _mm256_max_epu8(answer_8bit, result0);

			const __m256i index_score_matrix1_8bit = _mm256_add_epi8(sequence_tate1_x4, sequence_yoko);
			const __m256i value_score_matrix1_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix1_8bit);
			const __m256i tmp4y1 = _mm256_max_epu8(prev1, result0);
			const __m256i tmp5y1 = _mm256_adds_epu8(prev0, value_score_matrix1_plus_gap_8bit);
			const __m256i tmp6y1 = _mm256_max_epu8(tmp5y1, tmp4y1);
			const __m256i result1 = _mm256_subs_epu8(tmp6y1, gap_8bit);
			answer_8bit = _mm256_max_epu8(answer_8bit, result1);


			yoko[j - 2] = prev12;
			prev12 = prev1;
			prev1 = result1;
			prev02 = prev0;
			prev0 = result0;
			naname = ue;

		}
		yoko[128] = prev12;
		yoko[129] = prev1;
	}

	alignas(32)uint8_t answers[32] = {};
	_mm256_storeu_si256((__m256i *)answers, answer_8bit);

	for (int i = 0; i < 32; ++i)dest[i] = answers[i];
	return answers[0];
}

int SmithWaterman_8b111x32mark3(
	const std::array<uint8_t, 128 * 32>&obs1,
	const std::array<uint8_t, 128>&obs2,
	std::array<int, 32>& dest) {

	//上のmark2からの変更点：
	//(1)最初のバイト行列転置をAVX2で高速化した。
	//   参考: https://qiita.com/beru/items/12b4249c95a012a28ccd


	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;

	alignas(32)uint8_t transposed_obs1[32 * 128];
	//for (int i = 0; i < 128; ++i)for (int j = 0; j < 32; ++j)transposed_obs1[i * 32 + j] = obs1[j * 128 + i];

#define TRANSPOSE_16_16(ii,jj) \
do{\
	__m256i a0 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0x8 * 128]), (__m128i *)(&obs1[ii + 0x0 * 128]));\
	__m256i a1 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0x9 * 128]), (__m128i *)(&obs1[ii + 0x1 * 128]));\
	__m256i a2 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xA * 128]), (__m128i *)(&obs1[ii + 0x2 * 128]));\
	__m256i a3 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xB * 128]), (__m128i *)(&obs1[ii + 0x3 * 128]));\
	__m256i a4 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xC * 128]), (__m128i *)(&obs1[ii + 0x4 * 128]));\
	__m256i a5 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xD * 128]), (__m128i *)(&obs1[ii + 0x5 * 128]));\
	__m256i a6 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xE * 128]), (__m128i *)(&obs1[ii + 0x6 * 128]));\
	__m256i a7 = _mm256_loadu2_m128i((__m128i *)(&obs1[ii + 0xF * 128]), (__m128i *)(&obs1[ii + 0x7 * 128]));\
	__m256i b0 = _mm256_unpacklo_epi8(a0, a1);\
	__m256i b1 = _mm256_unpacklo_epi8(a2, a3);\
	__m256i b2 = _mm256_unpacklo_epi8(a4, a5);\
	__m256i b3 = _mm256_unpacklo_epi8(a6, a7);\
	__m256i b4 = _mm256_unpackhi_epi8(a0, a1);\
	__m256i b5 = _mm256_unpackhi_epi8(a2, a3);\
	__m256i b6 = _mm256_unpackhi_epi8(a4, a5);\
	__m256i b7 = _mm256_unpackhi_epi8(a6, a7);\
	a0 = _mm256_unpacklo_epi16(b0, b1);\
	a1 = _mm256_unpacklo_epi16(b2, b3);\
	a2 = _mm256_unpackhi_epi16(b0, b1);\
	a3 = _mm256_unpackhi_epi16(b2, b3);\
	a4 = _mm256_unpacklo_epi16(b4, b5);\
	a5 = _mm256_unpacklo_epi16(b6, b7);\
	a6 = _mm256_unpackhi_epi16(b4, b5);\
	a7 = _mm256_unpackhi_epi16(b6, b7);\
	b0 = _mm256_unpacklo_epi32(a0, a1);\
	b1 = _mm256_unpackhi_epi32(a0, a1);\
	b2 = _mm256_unpacklo_epi32(a2, a3);\
	b3 = _mm256_unpackhi_epi32(a2, a3);\
	b4 = _mm256_unpacklo_epi32(a4, a5);\
	b5 = _mm256_unpackhi_epi32(a4, a5);\
	b6 = _mm256_unpacklo_epi32(a6, a7);\
	b7 = _mm256_unpackhi_epi32(a6, a7);\
	a0 = _mm256_permute4x64_epi64(b0, _MM_SHUFFLE(3, 1, 2, 0));\
	a1 = _mm256_permute4x64_epi64(b1, _MM_SHUFFLE(3, 1, 2, 0));\
	a2 = _mm256_permute4x64_epi64(b2, _MM_SHUFFLE(3, 1, 2, 0));\
	a3 = _mm256_permute4x64_epi64(b3, _MM_SHUFFLE(3, 1, 2, 0));\
	a4 = _mm256_permute4x64_epi64(b4, _MM_SHUFFLE(3, 1, 2, 0));\
	a5 = _mm256_permute4x64_epi64(b5, _MM_SHUFFLE(3, 1, 2, 0));\
	a6 = _mm256_permute4x64_epi64(b6, _MM_SHUFFLE(3, 1, 2, 0));\
	a7 = _mm256_permute4x64_epi64(b7, _MM_SHUFFLE(3, 1, 2, 0));\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0x1 * 32]), (__m128i *)(&transposed_obs1[jj + 0x0 * 32]), a0);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0x3 * 32]), (__m128i *)(&transposed_obs1[jj + 0x2 * 32]), a1);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0x5 * 32]), (__m128i *)(&transposed_obs1[jj + 0x4 * 32]), a2);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0x7 * 32]), (__m128i *)(&transposed_obs1[jj + 0x6 * 32]), a3);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0x9 * 32]), (__m128i *)(&transposed_obs1[jj + 0x8 * 32]), a4);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0xB * 32]), (__m128i *)(&transposed_obs1[jj + 0xA * 32]), a5);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0xD * 32]), (__m128i *)(&transposed_obs1[jj + 0xC * 32]), a6);\
	_mm256_storeu2_m128i((__m128i *)(&transposed_obs1[jj + 0xF * 32]), (__m128i *)(&transposed_obs1[jj + 0xE * 32]), a7);\
}while(0)
	for (int i = 0; i < 128; i += 16)for (int j = 0; j < 32; j += 16) {
		TRANSPOSE_16_16((j * 128 + i), (i * 32 + j));
	}

#undef TRANSPOSE_16_16

	__m256i answer_8bit = _mm256_setzero_si256();
	const __m256i gap_8bit = _mm256_set1_epi8(GAP);
	const __m256i scorematrix_plus_gap_8bit = _mm256_set_epi8(
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH,
		GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH, GAP - MISMATCH,
		GAP - MISMATCH, GAP - MISMATCH, GAP - MISMATCH, GAP + MATCH);//MISMATCH <= GAPだからこれでよい。

	__m256i yoko[130];
	for (int i = 0; i < 130; ++i)yoko[i] = _mm256_setzero_si256();


	for (int i = 0; i < 128; i += 2) {
		__m256i prev0 = _mm256_setzero_si256();
		__m256i prev02 = _mm256_setzero_si256();
		__m256i prev1 = _mm256_setzero_si256();
		__m256i prev12 = _mm256_setzero_si256();
		const __m256i sequence_tate0_x4 = _mm256_slli_epi64(_mm256_loadu_si256((__m256i *)&transposed_obs1[(i + 0) * 32]), 2);
		const __m256i sequence_tate1_x4 = _mm256_slli_epi64(_mm256_loadu_si256((__m256i *)&transposed_obs1[(i + 1) * 32]), 2);

		__m256i naname = yoko[1];
		__m256i ue;

		for (int j = 2; j < 130; ++j) {

			ue = yoko[j];

			const __m256i sequence_yoko = _mm256_set1_epi8(obs2[j - 2]);

			const __m256i index_score_matrix0_8bit = _mm256_add_epi8(sequence_tate0_x4, sequence_yoko);
			const __m256i value_score_matrix0_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix0_8bit);
			const __m256i tmp4y0 = _mm256_max_epu8(prev0, ue);
			const __m256i tmp5y0 = _mm256_adds_epu8(naname, value_score_matrix0_plus_gap_8bit);
			const __m256i tmp6y0 = _mm256_max_epu8(tmp5y0, tmp4y0);
			const __m256i result0 = _mm256_subs_epu8(tmp6y0, gap_8bit);
			answer_8bit = _mm256_max_epu8(answer_8bit, result0);

			const __m256i index_score_matrix1_8bit = _mm256_add_epi8(sequence_tate1_x4, sequence_yoko);
			const __m256i value_score_matrix1_plus_gap_8bit = _mm256_shuffle_epi8(scorematrix_plus_gap_8bit, index_score_matrix1_8bit);
			const __m256i tmp4y1 = _mm256_max_epu8(prev1, result0);
			const __m256i tmp5y1 = _mm256_adds_epu8(prev0, value_score_matrix1_plus_gap_8bit);
			const __m256i tmp6y1 = _mm256_max_epu8(tmp5y1, tmp4y1);
			const __m256i result1 = _mm256_subs_epu8(tmp6y1, gap_8bit);
			answer_8bit = _mm256_max_epu8(answer_8bit, result1);


			yoko[j - 2] = prev12;
			prev12 = prev1;
			prev1 = result1;
			prev02 = prev0;
			prev0 = result0;
			naname = ue;

		}
		yoko[128] = prev12;
		yoko[129] = prev1;
	}

	alignas(32)uint8_t answers[32] = {};
	_mm256_storeu_si256((__m256i *)answers, answer_8bit);

	for (int i = 0; i < 32; ++i)dest[i] = answers[i];
	return answers[0];
}

int unpack(const std::array<uint8_t, 32>&src, std::array<uint8_t, 128>&dest) {
	for (int i = 0; i < 32; ++i)for (int j = 0; j < 4; ++j)dest[i * 4 + j] = (src[i] >> (j * 2)) & 0b11;
	return dest[0];
}

int unpack_simd(const std::array<uint8_t, 32>&src, std::array<uint8_t, 128>&dest) {

	for (int half = 0; half < 32; half += 16) {

		const __m256i src_ymm = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i *)&src[half]));
		{
			__m256i answer = _mm256_setzero_si256();
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x80808004, 0x80808004, 0x80808004, 0x80808004, 0x80808000, 0x80808000, 0x80808000, 0x80808000));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x80808005, 0x80808005, 0x80808005, 0x80808005, 0x80808001, 0x80808001, 0x80808001, 0x80808001));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x80808006, 0x80808006, 0x80808006, 0x80808006, 0x80808002, 0x80808002, 0x80808002, 0x80808002));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x80808007, 0x80808007, 0x80808007, 0x80808007, 0x80808003, 0x80808003, 0x80808003, 0x80808003));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			_mm256_storeu_si256((__m256i *)&dest[half * 4], answer);
		}
		{
			__m256i answer = _mm256_setzero_si256();
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x8080800C, 0x8080800C, 0x8080800C, 0x8080800C, 0x80808008, 0x80808008, 0x80808008, 0x80808008));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x8080800D, 0x8080800D, 0x8080800D, 0x8080800D, 0x80808009, 0x80808009, 0x80808009, 0x80808009));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x8080800E, 0x8080800E, 0x8080800E, 0x8080800E, 0x8080800A, 0x8080800A, 0x8080800A, 0x8080800A));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			{
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, _mm256_set_epi32(0x8080800F, 0x8080800F, 0x8080800F, 0x8080800F, 0x8080800B, 0x8080800B, 0x8080800B, 0x8080800B));
				const __m256i a1 = _mm256_srlv_epi32(a0, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0));
				const __m256i a2 = _mm256_and_si256(a1, _mm256_set1_epi32(0b11));
				const __m256i a3 = _mm256_shuffle_epi8(a2, _mm256_set_epi32(0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080));
				answer = _mm256_or_si256(answer, a3);
			}
			_mm256_storeu_si256((__m256i *)&dest[half * 4 + 32], answer);
		}
	}
	return dest[0];
}
int unpack_simd2(const std::array<uint8_t, 32>&src, std::array<uint8_t, 128>&dest) {

	const __m256i shifter = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
	const __m256i bitmask = _mm256_set1_epi32(0b11);
	const __m256i before_shuffle_shifter = _mm256_set1_epi32(1);
	const __m256i meta_after_shuffle_mask = _mm256_set_epi32(0x0B0A0908, 0x07060504, 0x03020100, 0x0F0E0D0C, 0x0B0A0908, 0x07060504, 0x03020100, 0x0F0E0D0C);

	for (int half = 0; half < 32; half += 16) {

		const __m256i src_ymm = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i *)&src[half]));
		{
			__m256i answer = _mm256_setzero_si256();
			__m256i before_shuffle_mask = _mm256_set_epi32(0x80808004, 0x80808004, 0x80808004, 0x80808004, 0x80808000, 0x80808000, 0x80808000, 0x80808000);
			__m256i after_shuffle_mask = _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400);
			for (int i = 0; i < 4; ++i) {
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, before_shuffle_mask);
				const __m256i a1 = _mm256_srlv_epi32(a0, shifter);
				const __m256i a2 = _mm256_and_si256(a1, bitmask);
				const __m256i a3 = _mm256_shuffle_epi8(a2, after_shuffle_mask);
				answer = _mm256_or_si256(answer, a3);
				before_shuffle_mask = _mm256_add_epi32(before_shuffle_mask, before_shuffle_shifter);
				after_shuffle_mask = _mm256_shuffle_epi8(after_shuffle_mask, meta_after_shuffle_mask);
			}
			_mm256_storeu_si256((__m256i *)&dest[half * 4], answer);
		}
		{
			__m256i answer = _mm256_setzero_si256();
			__m256i before_shuffle_mask = _mm256_set_epi32(0x8080800C, 0x8080800C, 0x8080800C, 0x8080800C, 0x80808008, 0x80808008, 0x80808008, 0x80808008);
			__m256i after_shuffle_mask = _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400);
			for (int i = 0; i < 4; ++i) {
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, before_shuffle_mask);
				const __m256i a1 = _mm256_srlv_epi32(a0, shifter);
				const __m256i a2 = _mm256_and_si256(a1, bitmask);
				const __m256i a3 = _mm256_shuffle_epi8(a2, after_shuffle_mask);
				answer = _mm256_or_si256(answer, a3);
				before_shuffle_mask = _mm256_add_epi32(before_shuffle_mask, before_shuffle_shifter);
				after_shuffle_mask = _mm256_shuffle_epi8(after_shuffle_mask, meta_after_shuffle_mask);
			}
			_mm256_storeu_si256((__m256i *)&dest[half * 4 + 32], answer);
		}
	}
	return dest[0];
}
int unpack_simd3(const std::array<uint8_t, 32>&src, std::array<uint8_t, 128>&dest) {

	const __m256i shifter = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
	const __m256i bitmask = _mm256_set1_epi32(0b11);
	const __m256i before_shuffle_shifter = _mm256_set1_epi32(1);
	const __m256i meta_after_shuffle_mask = _mm256_set_epi32(0x0B0A0908, 0x07060504, 0x03020100, 0x0F0E0D0C, 0x0B0A0908, 0x07060504, 0x03020100, 0x0F0E0D0C);

	for (int half = 0; half < 32; half += 16) {

		const __m256i src_ymm = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i *)&src[half]));
		{
			__m256i answer1 = _mm256_setzero_si256();
			__m256i answer2 = _mm256_setzero_si256();
			__m256i before_shuffle_mask1 = _mm256_set_epi32(0x80808004, 0x80808004, 0x80808004, 0x80808004, 0x80808000, 0x80808000, 0x80808000, 0x80808000);
			__m256i before_shuffle_mask2 = _mm256_set_epi32(0x8080800C, 0x8080800C, 0x8080800C, 0x8080800C, 0x80808008, 0x80808008, 0x80808008, 0x80808008);
			__m256i after_shuffle_mask = _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400);
			for (int i = 0; i < 4; ++i) {
				const __m256i a0 = _mm256_shuffle_epi8(src_ymm, before_shuffle_mask1);
				const __m256i b0 = _mm256_shuffle_epi8(src_ymm, before_shuffle_mask2);
				const __m256i a1 = _mm256_srlv_epi32(a0, shifter);
				const __m256i b1 = _mm256_srlv_epi32(b0, shifter);
				const __m256i a2 = _mm256_and_si256(a1, bitmask);
				const __m256i b2 = _mm256_and_si256(b1, bitmask);
				const __m256i a3 = _mm256_shuffle_epi8(a2, after_shuffle_mask);
				const __m256i b3 = _mm256_shuffle_epi8(b2, after_shuffle_mask);
				answer1 = _mm256_or_si256(answer1, a3);
				answer2 = _mm256_or_si256(answer2, b3);
				before_shuffle_mask1 = _mm256_add_epi32(before_shuffle_mask1, before_shuffle_shifter);
				before_shuffle_mask2 = _mm256_add_epi32(before_shuffle_mask2, before_shuffle_shifter);
				after_shuffle_mask = _mm256_shuffle_epi8(after_shuffle_mask, meta_after_shuffle_mask);
			}
			_mm256_storeu_si256((__m256i *)&dest[half * 4], answer1);
			_mm256_storeu_si256((__m256i *)&dest[half * 4 + 32], answer2);
		}
	}
	return dest[0];
}

int SemiGlobal_111(
	const std::array<uint8_t, 16384>&obs1,
	const std::array<uint8_t, 16384>&obs2) {

	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1;
	constexpr int minus_inf = std::numeric_limits<int>::min() / 2;

	std::vector<int>dp((16384 + 1) * (16384 + 1), minus_inf);
	dp[0] = 0;

#define INDEX(ii, jj) ((ii) * (16384 + 1) + (jj))

	int answer = 0;

	for (int i = 0; i <= 16384; ++i) {
		for (int j = 0; j <= 16384; ++j) {
			const int index = INDEX(i, j);
			if (i && j)dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 1)] + (obs1[i - 1] == obs2[j - 1] ? MATCH : -MISMATCH));
			if (i)dp[index] = std::max<int>(dp[index], dp[INDEX(i - 1, j - 0)] - GAP);
			if (j)dp[index] = std::max<int>(dp[index], dp[INDEX(i - 0, j - 1)] - GAP);
			answer = std::max<int>(answer, dp[index]);
			//std::cout << dp[index] << " ";
		}
		//std::cout << std::endl;
	}

#undef INDEX

	return answer;
}

int SemiGlobal_AdaptiveBanded_16kLength_11170(
	const std::array<uint8_t, 16384>&obs1,
	const std::array<uint8_t, 16384>&obs2) {

	//SemiGlobalと言ってるのは、
	//・ゼロとの比較をしない (Global)
	//・左上端からアライメントがスタートする (Global)
	//・右下端でアライメントが終わるとは限らない。スコア最大の地点からトレースバックする (Local)
	//の意味

	//TODO: 16384決め打ちだとデカすぎてコレ自体をテストしにくいので、決め打ちしないようにしよう←ゴリ押せる程度のデカさだった

	//TODO: 全埋めと座標系が微妙に違ってて気持ち悪いので修正する

	//TODO: 似ている配列に関して全埋めとスコアが一致することを確認する
	//TODO: トレースバックを実装する
	//TODO: 似ている配列に関して全埋めとトレースバック結果が一致することを確認する


	constexpr uint8_t MATCH = 1, MISMATCH = 1, GAP = 1, X_THRESHOLD = 70;
	constexpr int minus_inf = std::numeric_limits<int>::min() / 2;

	std::map<int64_t, int>dp;
	const auto Get = [&](const int64_t y, const int64_t x) {
		assert(0 <= y && y < (16384 * 2) && 0 <= x && x < (16384 * 2));
		if (y == 0 && x == 0)return 0;
		if (y == 0)return int(-x * GAP);
		if (x == 0)return int(-y * GAP);

		const int64_t index = (y * (16384 * 2)) + x;
		if (dp.find(index) == dp.end())return minus_inf;
		return dp[index];
	};
	const auto Set = [&](const int64_t y, const int64_t x, const int value) {
		assert(0 <= y && y < (16384 * 2) && 0 <= x && x < (16384 * 2));
		dp[(y * (16384 * 2)) + x] = value;
	};

	int max_pos_y = 0, max_pos_x = 0, max_score = 0;

	//最初に左上の三角形部分を埋める。
	for (int y = 0; y < 32; ++y)for (int x = 0; x < 32 - y; ++x) {
		if (y == 0 && x == 0) {
			Set(0, 0, 0);
			continue;
		}
		int score = minus_inf;
		if (y&&x)score = std::max<int>(score, Get(y - 1, x - 1) + (obs1[y - 1] == obs2[x - 1] ? MATCH : -MISMATCH));
		if (y)score = std::max<int>(score, Get(y - 1, x) - GAP);
		if (x)score = std::max<int>(score, Get(y, x - 1) - GAP);
		Set(y, x, score);
		if (max_score < score) {
			max_pos_y = y;
			max_pos_x = x;
			max_score = score;
		}
	}

	//この時点で、(0,31)～(31,0)までの斜め区間の32要素と、そこから左上の区間が求まっている。
	//以降の繰り返し手順は、
	//(1)下に行くか右に行くか決める
	//(2)決めた方向の32要素を計算する
	//で、この他にX-dropの打ち切り基準の計算とかもある。

	for (int y_now = 0, x_now = 31, center_max_score = Get(y_now + 16, x_now - 16); y_now <= 16384 + 32 && x_now <= 16384 + 32;) {

		//X-drop
		const int center_now_score = Get(y_now + 16, x_now - 16);
		if (center_now_score + X_THRESHOLD < center_max_score) {
			break;
		}
		else if (center_max_score < center_now_score)center_max_score = center_now_score;

		const int now_upperleft_score = Get(y_now, x_now);
		const int now_lowerright_score = Get(y_now + 31, x_now - 31);
		if (now_upperleft_score < now_lowerright_score) {
			//下に行く
			for (int i = 0; i < 32; ++i) {
				const int y_next = y_now + i + 1;
				const int x_next = x_now - i;
				int score = minus_inf;
				if (y_next && x_next && y_next <= 16384 && x_next <= 16384)score = std::max<int>(score, Get(y_next - 1, x_next - 1) + (obs1[y_next - 1] == obs2[x_next - 1] ? MATCH : -MISMATCH));
				if (y_next)score = std::max<int>(score, Get(y_next - 1, x_next) - GAP);
				if (x_next)score = std::max<int>(score, Get(y_next, x_next - 1) - GAP);
				Set(y_next, x_next, score);
				if (max_score < score) {
					max_pos_y = y_next;
					max_pos_x = x_next;
					max_score = score;
				}
			}
			++y_now;
		}
		else {
			//右に行く
			for (int i = 0; i < 32; ++i) {
				const int y_next = y_now + i;
				const int x_next = x_now - i + 1;
				int score = minus_inf;
				if (y_next && x_next && y_next <= 16384 && x_next <= 16384)score = std::max<int>(score, Get(y_next - 1, x_next - 1) + (obs1[y_next - 1] == obs2[x_next - 1] ? MATCH : -MISMATCH));
				if (y_next)score = std::max<int>(score, Get(y_next - 1, x_next) - GAP);
				if (x_next)score = std::max<int>(score, Get(y_next, x_next - 1) - GAP);
				Set(y_next, x_next, score);
				if (max_score < score) {
					max_pos_y = y_next;
					max_pos_x = x_next;
					max_score = score;
				}
			}
			++x_now;
		}
	}

	return max_score;
}

void TestSemiGlobal() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);
	std::uniform_int_distribution<int> dice(0, 19);

	for (int iteration = 0; iteration < 10000000; iteration++) {
		std::cout << iteration << std::endl;
		std::array<uint8_t, 16384>a, b;
		for (int i = 0; i < 16384; ++i) {
			a[i] = dna(rnd);
			if (dice(rnd))b[i] = a[i];
			else b[i] = dna(rnd);
		}

		const int ans2 = SemiGlobal_AdaptiveBanded_16kLength_11170(a, b);
		const int ans1 = SemiGlobal_111(a, b);

		std::cout << "ans1 = " << ans1 << std::endl;
		std::cout << "ans2 = " << ans2 << std::endl;

		assert(ans1 == ans2);
	}
	return;

}


void TestUnpack() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dist(0, 255);

	for (int iteration = 0; iteration < 10000000; iteration++) {
		std::cout << iteration << std::endl;
		std::array<uint8_t, 32>a;
		for (int i = 0; i < 32; ++i)a[i] = dist(rnd);
		std::array<uint8_t, 128>b1 = std::array<uint8_t, 128>();
		std::array<uint8_t, 128>b2 = std::array<uint8_t, 128>();
		std::array<uint8_t, 128>b3 = std::array<uint8_t, 128>();
		std::array<uint8_t, 128>b4 = std::array<uint8_t, 128>();

		unpack(a, b1);
		unpack_simd(a, b2);
		unpack_simd2(a, b3);
		unpack_simd2(a, b4);
		for (int i = 0; i < 128; ++i)assert(b1[i] == b2[i]);
		for (int i = 0; i < 128; ++i)assert(b1[i] == b3[i]);
		for (int i = 0; i < 128; ++i)assert(b1[i] == b4[i]);
	}
	return;
}

void speedtestunpack() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dist(0, 255);
	std::array<uint8_t, 32>a;
	for (int i = 0; i < 32; ++i)a[i] = dist(rnd);
	std::array<uint8_t, 128>b = std::array<uint8_t, 128>();

	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 10000000; ++iteration) {
			volatile int score = unpack(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "normal version: " << elapsed << " ms / 10M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 100000000; ++iteration) {
			volatile int score = unpack_simd(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd version: " << elapsed << " ms / 100M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 100000000; ++iteration) {
			volatile int score = unpack_simd2(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd2 version: " << elapsed << " ms / 100M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 100000000; ++iteration) {
			volatile int score = unpack_simd3(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd3 version: " << elapsed << " ms / 100M" << std::endl;
	}
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
		const int ans6 = SmithWaterman_simd5(a, b, score_matrix, gap_penalty);
		const int ans7 = SmithWaterman_simd6(a, b, score_matrix, gap_penalty);
		const int ans8 = SmithWaterman_simd7(a, b, score_matrix, gap_penalty);
		const int ans9 = SmithWaterman_simd8(a, b, score_matrix, gap_penalty);
		const int ans10 = SmithWaterman_simd9(a, b, score_matrix, gap_penalty);
		assert(ans1 == ans2);
		assert(ans1 == ans3);
		assert(ans1 == ans4);
		assert(ans1 == ans5);
		assert(ans1 == ans6);
		assert(ans1 == ans7);
		assert(ans1 == ans8);
		assert(ans1 == ans9);
		assert(ans1 == ans10);
	}
	return;
}

void TestSimdSmithWaterman111() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	for (int iteration = 0; iteration < 10000000; iteration++) {
		std::cout << iteration << std::endl;
		std::array<uint8_t, 128>a, b;
		for (int i = 0; i < 128; ++i) {
			a[i] = dna(rnd);
			b[i] = dna(rnd);
		}

		const int ans1 = SmithWaterman_111(a, b);
		const int ans2 = SmithWaterman_8bit111simd(a, b);
		assert(ans1 == ans2);
	}
	return;
}

void TestSimdSmithWaterman111x32() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	for (int iteration = 0; iteration < 10000000; iteration++) {
		std::cout << iteration << std::endl;
		std::array<uint8_t, 128 * 32>a;
		std::array<uint8_t, 128>b;
		std::array<int, 32>dest;
		std::array<int, 32>ref_ans;
		for (int i = 0; i < 128 * 32; ++i)a[i] = dna(rnd);
		for (int i = 0; i < 128; ++i)b[i] = dna(rnd);

		for (int i = 0; i < 32; ++i) {
			std::array<uint8_t, 128>aa;
			for (int j = 0; j < 128; ++j)aa[j] = a[i * 128 + j];
			const int ans1 = SmithWaterman_111(aa, b);
			ref_ans[i] = ans1;
		}
		const int ans2 = SmithWaterman_8b111x32mark1(a, b, dest);
		for (int i = 0; i < 32; ++i)assert(ref_ans[i] == dest[i]);
		const int ans3 = SmithWaterman_8b111x32mark2(a, b, dest);
		for (int i = 0; i < 32; ++i)assert(ref_ans[i] == dest[i]);
		const int ans4 = SmithWaterman_8b111x32mark3(a, b, dest);
		for (int i = 0; i < 32; ++i)assert(ref_ans[i] == dest[i]);
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
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd5(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd5 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd6(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd6 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd7(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd7 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd8(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd8 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd9(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd9 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_8bit111simd(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "SmithWaterman_8bit111simd: " << elapsed << " ms / 1M" << std::endl;
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
			volatile int score = SmithWaterman_simd7(a, b, score_matrix, gap_penalty);
		}
	}
	return;
}

void InfinitySW111x32() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	std::array<uint8_t, 128 * 32>a;
	std::array<uint8_t, 128>b;
	std::array<int, 32>dest;
	for (int i = 0; i < 128 * 32; ++i)a[i] = dna(rnd);
	for (int i = 0; i < 128; ++i)b[i] = dna(rnd);
	{
		for (;;) {
			volatile int score = SmithWaterman_8b111x32mark1(a, b, dest);
		}
	}
	return;
}

void speedtest111x32() {
	std::mt19937_64 rnd(10000);
	std::uniform_int_distribution<int> dna(0, 3);

	std::array<uint8_t, 128 * 32>aa;
	std::array<uint8_t, 128>a, b;
	std::array<int, 32>dest;
	for (int i = 0; i < 128 * 32; ++i)aa[i] = dna(rnd);
	for (int i = 0; i < 128; ++i) {
		a[i] = dna(rnd);
		b[i] = dna(rnd);
	}

	std::array<int8_t, 16>score_matrix = {
		1,-1,-1,-1,
		-1,1,-1,-1,
		-1,-1,1,-1,
		-1,-1,-1,1 };
	uint8_t gap_penalty = 1;

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
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd7(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd7 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_simd9(a, b, score_matrix, gap_penalty);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd9 version: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 1000000; ++iteration) {
			volatile int score = SmithWaterman_8bit111simd(a, b);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd8bit111: " << elapsed << " ms / 1M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 10000000 / 32; ++iteration) {
			volatile int score = SmithWaterman_8b111x32mark1(aa, b, dest);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd8bit111 x32 mark1: " << elapsed << " ms / 10M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 10000000 / 32; ++iteration) {
			volatile int score = SmithWaterman_8b111x32mark2(aa, b, dest);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd8bit111 x32 mark2: " << elapsed << " ms / 10M" << std::endl;
	}
	{
		auto start = std::chrono::system_clock::now(); // 計測開始時間
		for (int iteration = 0; iteration < 10000000 / 32; ++iteration) {
			volatile int score = SmithWaterman_8b111x32mark3(aa, b, dest);
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
		std::cout << "simd8bit111 x32 mark3: " << elapsed << " ms / 10M" << std::endl;
	}
	return;
}

int main(void) {

	TestSemiGlobal();

	//TestUnpack();
	//speedtestunpack();
	//speedtestunpack();
	//speedtestunpack();

	//TestSimdSmithWaterman111x32();
	//TestSimdSmithWaterman111();
	//TestSimdSmithWaterman();
	//InfinitySW111x32();
	//InfinitySW();
	//speedtest111x32();
	//speedtest111x32();
	//speedtest111x32();
	//SpeedTest();
	//SpeedTest();
	//SpeedTest();

	return 0;
}
