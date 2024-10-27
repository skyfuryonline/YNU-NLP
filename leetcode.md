## 中级算法
[中级算法](https://leetcode.cn/leetbook/detail/top-interview-questions-medium/)

[1.两数之和](https://leetcode.cn/problems/two-sum/description/)
思路：
总结c++中几种map(unordered_map,multimap,map)的区别和使用方法；
```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int>ump;
        int len = nums.size();
        int i=0;
        vector<int> v;
        for(;i<len;i++){
            int s = target-nums[i];
            auto it = ump.find(s);
            if(it!=ump.end() && it->second!=i){
            //防止出现{3},6的情况，6-3=3，但是只有一个元素
                v.push_back(i);
                v.push_back(it->second);
                break;
            }else{
                ump.insert({nums[i],i});
            }
        }
        return v;
    }
};
```

[三数之和](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvpj16/)
解法概述：假设数组有序，使用双指针，固定其中一个数，即可对剩余的有序数组进行双指针。对于重复数据需要过滤，可以考虑(a1,a2,b,c),如果已经有一组a1bc满足，则判断下一个a2是否等于前一个a1，如果等则跳过。
```c++
class Solution {

public:

vector<vector<int>> threeSum(vector<int>& nums) {

int s = nums.size();

sort(nums.begin(),nums.end());

vector<vector<int> >ans;

for(int left=0;left<s;left++){//left是遍历整个数组的指针

if(nums[left]>0) break;//排过序，正数之和不可能为负

if(left>0 && nums[left]==nums[left-1])//从第二个开始如果和前一个相同则去重跳过,因为前一个已经借助左右指针判断过一遍了

continue;

int i=left+1,j=s-1;//左右指针进行判断

while(i<j){

int sum = nums[i]+nums[j];

if(sum==-nums[left]){

vector<int> tmp;tmp.push_back(nums[i]);

tmp.push_back(nums[j]);tmp.push_back(nums[left]);

ans.push_back(tmp);

while(i<j && nums[i]==nums[i+1])i++;
//去重，nums[i]+nums[j],跳过i右边与nums[i]相同的重复值

while(i<j && nums[j]==nums[j-1])j--;
//去重，nums[i]+nums[j],跳过j左边与nums[j]相同的重复值

i++;j--;
//经过上面两步移动后此时ij指向重复nums[i]或nums[j]的最后一个。
//i++和j++保证nums[i]+nums[j]==-nums[left]后的正常移动

}else if(sum<-nums[left]) i++;
else j--;
}
}
return ans;
}
};
```

[矩阵置零](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvmy42/)
解法概述：
O(mn)额外空间解法分析：按行遍历，将每行中为0的列号存入map，即(key=行号从1～m，value=元素为0列号集合)，最后遍历map进行对应元素归零即可。
```c++
class Solution {

public:

void setZeroes(vector<vector<int>>& matrix) {

int m =matrix.size();//行

int n =matrix[0].size();//列

map<int,vector<int> > check;//保存每行为零元素的列号集合

for(int i=0;i<m;i++){

for(int j=0;j<n;j++){

if(matrix[i][j]==0){

check[i].push_back(j);
}}}

map<int,vector<int> >::iterator it = check.begin();

while(it!=check.end()){//遍历字典开始归零

for(int j=0;j<n;j++)
matrix[it->first][j] = 0;//先将这行置零

int s = it->second.size();//某一行为零元素的列号个数

for(int count=0;count<s;count++){
for(int i=0;i<m;i++)
matrix[i][it->second[count]] = 0;//将为零元素所在列归零
}
it++;

}}};
```

O(m+n)空间的解法分析：两个临时数组记录行和列信息。一个元素，如果其行列在数组中被标记，则修改为0，否则检查该元素是否为0并修改标记数组。数组可使用bool值。注意区分bool(4)(4)和col(4),row(4)的区别，一个使用4x4=16个空间，一个仅使用4+4=8个空间。

```c++
class Solution {

public:

void setZeroes(vector<vector<int>>& matrix) {

int m = matrix.size(),n = matrix[0].size();

bool row[m],col[n];//记录行列信息

memset(col,0,sizeof(col));

memset(row,0,sizeof(row));

for(int i=0;i<m;i++){
for(int j=0;j<n;j++){
if(matrix[i][j]==0){
row[i] = true;
col[j] = true;
}}}

  

for(int i=0;i<m;i++){
for(int j=0;j<n;j++){

if((row[i]==true||col[j]==true)){

matrix[i][j]=0;

}}}}};
```

常量空间解法分析：使用矩阵的第一行和第一列作为辅助数组，如果矩阵中有元素是零，则将其对应的辅助数组标记为零。因为如果有零元素，则对应行列都将设置为零，所以不必担心修改原矩阵数据的问题。从第二行和第二列开始遍历矩阵，所以先要记录原第一行第一列中是否有零元素。
```c++
class Solution {

public:

void setZeroes(vector<vector<int>>& matrix) {

int m = matrix.size(),n = matrix[0].size();

bool col_zero = false,row_zero = false;
//仅用于修改辅助数组，判断第一行和第一列是否需要归零
//不需要记住零元素的位置，因为不论有几个零，作为标记数组后续在对矩阵进行归零操作判断时看matrix[i][0]或matrix[0][j]即可。

for(int i=0;i<m;i++){
for(int j=0;j<n;j++){

if(matrix[i][j]==0){
if(i==0){
row_zero = true;
}if(j==0){
col_zero = true;
}
matrix[0][j] = 0;
matrix[i][0] = 0;//利用第一行和第一列进行标记

}}}

for(int i=1;i<m;i++){//先从辅助数组之外的其他位置开始判断
for(int j=1;j<n;j++){

if(matrix[i][0]==0||matrix[0][j]==0)

matrix[i][j] = 0;

}}

if(col_zero)
for(int i=0;i<m;i++)
matrix[i][0] = 0;

if(row_zero)
for(int j=0;j<n;j++)
matrix[0][j] = 0;

}};
```

[字母异位词分组](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvaszc/)
朴素解法概述：先将每个字符串按照字母大小升序排序，然后用map判断哪些字符串属于同一类。
```c++
class Solution {
public:
vector<vector<string>> groupAnagrams(vector<string>& strs) {
map<string,bool> check;
map<string,vector<string> > getw;
vector<string>::iterator it = strs.begin();
vector<vector<string> >ans;
while(it!=strs.end()){
string tem = mysort(*it);
if(check[tem]==false){
check[tem] = true;
getw[tem].push_back(*it);
}else {
getw[tem].push_back(*it);
}
it++;
}

map<string,vector<string> >::iterator itt = getw.begin();
while(itt!=getw.end()){
ans.push_back(itt->second);
itt++;
}
return ans;
}

string mysort(string tem){//自定义排序函数，对字符串中的字符升序排列
string temp = tem;
int len = temp.size();
for(int i=0;i<len;i++){
for(int j=i+1;j<len;j++){
if(temp[i]-'a'>temp[j]-'a'){
swap(temp[i],temp[j]);
}
}}
return temp;
}

void swap(char& a,char& b){//辅助函数，为排序进行字符交换
char tem = a;
a = b;
b = tem;}

};
```

使用auto并sort排序版本：
```c++
class Solution {
public:
vector<vector<string>> groupAnagrams(vector<string>& strs) {
map<string,vector<string> > check;

for(const auto &i:strs){
string tem = i;
sort(tem.begin(),tem.end());
check[tem].push_back(i);
}

vector<vector<string> >ans;

for(const auto &i:check){
ans.push_back(i.second);//总结调用方式,注意i.second与i->second
}
return ans;
}};
```

质数相乘法：
```
质数(素数)性质总结：
1.因数只有1和它本身，即质数不能被其他自然数整除,最小质数为2
2.质因数分解定理:每个合数都可以写成几个质数相乘的形式
3.唯一分解定理:任何一个大于1的自然数N，如果N不为质数，那么N可以唯一分解成有限个质数的乘积----不计次序的情况下分解方式唯一
4.如果一个数i可以表示成两因数相乘的形式，那么其中一个因数一定满足小于等于根号下i。在寻找一个数的所有质因数时，通过检查这个数是否小于它的平方根，可以有效地减少计算量，因为如果存在一个大于该数平方根的因数，那么必定存在一个小于或等于该数平方根的对应因数，从而避免了不必要的计算
!!5‌.埃氏筛素数(厄拉多塞筛法)-----必看
!!6.线性筛素数--------必看

补充求gcd(最大公因数)和lcm(最小公倍数)的方法：
设：
a = 54 = 2x3x3x3
b = 48 = 2x2x2x2x3
将ab如上式进行质数分解：
gcd = ab中共有的质因数的积
lcm = gcd * 两个数剩余质因数之积
//即短除法干的事，除出来的是公因数，相乘即gcd，再乘底下的结果即lcm

所以：
gcd = 2x3=6
lcm = 6 x 2x2x2x3x3 = 432

验证：
#include<numeric>
int a = 54,b = 48;
cout<<gcd(a,b)<<" "<<lcm(a,b);
//输出为：6 432

三个数的最小公倍数：
短除法要做到底下的数两两互质，然后再将三者的公因数乘上底下的数字；
```
质数(素数)题目：
[[#^f744c1|求质数]]
[[#^6a4251|回文质数]]

解法概述：每个字母代表一个质数，字母异位词相乘得到的结果相同。参考唯一分解定理
```c++
class Solution {

public:

vector<vector<string>> groupAnagrams(vector<string>& strs) {
int primeNum[26] ={5, 71, 31, 29, 2, 53, 59, 23, 11, 89,
79, 37, 41, 13, 7, 43, 97, 17, 19, 3, 47, 73, 61, 83, 67, 101};//26个素数对应26个字母

map<char,int> check;//建立字符和素数的字典

for(int i=0;i<26;i++)

check['a'+i] = primeNum[i];//对26个字母分配素数

map<unsigned int,vector<string> >m;
//！！一定要使用unsigned int类型，因为素数为正，且可能乘积极大，int只能到2^31-1,unsigned int可以到2^32-1.

for(const auto str:strs){//选择每一个字符串进行操作

unsigned int key = 1;//！！使用unsigned int，理由同上

int len = str.size();

for(const auto c:str){//对字符串的每一个字符操作

key *= check[c];

}
m[key].push_back(str);
}

vector<vector<string> > ans;

map<unsigned int,vector<string> >::iterator it = m.begin();
//！!使用unsigned int，理由同上
while(it!=m.end()){
ans.push_back(it->second);
it++;
}
return ans;
}};
```

[无重复字符的最长子串](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xv2kgi/)
双指针解法概述：使用ij控制一段无重复字符的最大子串。j遍历字符，同时用map添加/更新各个字符的位置。如果j遍历到的字符重复，则：将i跳到这个重复字符上次记录位置的下一个位置。维护一个maxLen变量作为最终符合要求返回的结果。
```c++
class Solution {

public:

int lengthOfLongestSubstring(string s) {

int len = s.size();

map<char,int> check;
//用一个字典保存每一个字符出现的位置，同时可以判断重复，如果j在遍历的过程中发现重复的字符，则i跳到这个重复字符上次记录的位置的下一个位置。

int maxLen = 0;

for(int i=0,j=0;j<len;j++){//j向后遍历，i控制无重复字符的最长子串

if(check.find(s[j])!=check.end()){
//注意总结map.find和set.find用法

i = max(i,check[s[j]]+1);//如上述，跳到合适的位置

}

check[s[j]]=j;//添加/更新这个新遍历到的字符的位置

maxLen = max(maxLen,j-i+1);//更新ij控制的符合条件的串的最大长度

}
return maxLen;
}};
```

[计数质数](https://leetcode.cn/problems/count-primes/description/)

^f744c1

解法概述：
1.朴素思想：用待验证的数n依次除以2～n-1,如果余数均不为0，则n为质数；
2.改进朴素思想：用n依次除以2~n/2(超过n/2的数不是n的倍数--1倍即为n，如果超过n/2,则2倍将大于n，故超过n/2的数不是n的倍数)
```
一个直观的例子：带测试的数字为368
1 x 368 = 368
2 x 184 = 368
4 x 92 = 368
8 x 46 = 368
16 x 23 = 368
-------------->对应3，√368≈19.2，超过这个数字则由于乘法交换律重复
23 x 16 = 368
46 x 8 = 368
92 x 4 = 368
184 x 2 = 368
-------------->对应2，即185～367的数都不用验证，因为均大于368/2=184
368 x 1 = 368--->大于n/2的数中有且仅有它本身可以被n整除，即1倍
```
3.再改进：如果一个数i可以表示成两因数相乘的形式，那么其中一个因数一定满足小于等于根号下i
```c++
//实现第三种改进后的算法
class Solution {
public:
int countPrimes(int n) {
int count = 0;
for(int i=2;i<n;i++)

count+=isPrime(i);

return count;
}

private:
bool isPrime(int num){
for(int i=2;i*i<=num;i++){//注意这种写法，i此时就是√num,不需要sqrt
if(num%i==0)return false;
}
return true;
}};
```
4.埃氏筛素数(厄拉多塞筛法)：
分析：如果x为质数，则其倍数2x，3x...均为合数。设定bool isPrime(i)表示i是否为质数，是则为1，反则为0。从小到大遍历每个数，若为质数，则将其倍数(2开始)标为0。当遍历到数x时，若是合数，则一定是某个小于x的质数y的整数倍(可以结合唯一分解定理)，即当我们遍历y时，x已经被标记为0。
优化理解：从x^2开始标记，2x,3x在x之前就已被2,3标记过。即从x的x倍开始，x的"小于x的数"倍已经被标记过。
再补充理解：即将素数x的若干倍数标记为合数，这个倍数可以被优化从x的x倍开始(小于x的数已经在遍历他们的时候就标记了，比方7的3倍已经在3的7倍处被标记)。
```c++
class Solution {
public:
int countPrimes(int n) {

vector<bool> isPrime(n,1);//初始化数组，n个1

int ans=0;
for(int i=2;i<n;i++){//i是待测数字

if(isPrime[i]==1){
	ans++;
	if((long long)i*i<n){//因为从i^i开始，防止i^i超过限定个数
		for(int j=i*i;j<n;j+=i){
		//从i的i倍开始标记，i的i-1倍和i的i倍之间差i，所以步长为j+=i
		//另一种直观理解：倍数就是加自己，3倍就是加3次自己
			isPrime[j] = 0;
			}
		}
	}
}
return ans;
}};
```
5.线性筛素数：
分析：对埃氏筛素数的改进，埃氏筛素数存在冗余操作，如45这个合数会被3和5两个数标记(3的15倍和5的9倍)，线性筛素数让每个合数只被标记一次。时间复杂度为O(n).每个合数只被最小质因数标记。
算法概述：相比埃氏筛素数，多维护一个primes数组存储遇到的素数。且标记过程不再仅当x为素数时进行。对于整数x，不再标记x的所有倍数,而是只标记质数集合中的数与x相乘的数，且在发现x mod prime_i=0时结束标记。

为什么x mod prime_i=0时结束标记：x mod prime_i=0,可以写成：x = k\*prime_i;假设不结束，则x\*prime_i+1 = k\*prime_i\*prime_i+1,调整顺序后：k\*prime_i+1\*prime_i,此时x\*prime_i+1的最小质因数是prime_i，而不是prime_i+1,所以要在x mod prime_i=0的时候结束标记。（要让合数只被最小质因数标记）

一个例子：假设质数数组:(2,3..),待测数字12，12x2=24，所以标记24，即24的最小质因数是2，12mod2=0，所以停止；若不停：12x3=36，而12=2x6，故36的最小质因数应该为2，即2x18=36.
```c++
class Solution {
public:

int countPrimes(int n) {

vector<int> primes;//维护一个数组，存储当前得到的质数集合

vector<int> isPrime(n,1);

for(int i=2;i<n;i++){
	if(isPrime[i]){
		primes.push_back(i);
	}

for(int j=0;j<primes.size() && i*primes[j]<n;j++){
	isPrime[i*primes[j]]=0;
	if(i % primes[j]==0)
		break;

}}
return primes.size();
}};
```


[回文质数](https://leetcode.cn/problems/prime-palindrome/description/)

^6a4251

解法概述：先检查素数再检查回文，注意不存在长度为8的素数。也可以打表，先将200000000内的素数算出，后查表即可。
```c++
class Solution {
public:
int primePalindrome(int n) {
for(int i=n;;i++){
	if(i<pow(10,8) && i>=pow(10,7)){
	//不存在长度为8的素数
		i=pow(10,8);
}
if(isPalindrome(i)&&isPrime(i))
return i;

}}

private:
bool isPrime(int num){
if(num<2) return false;//注意细节

for(int i=2;i*i<=num;i++){
	if(num%i==0) return false;
}
return true;
}

bool isPalindrome(int num){
	string tem = to_string(num);
	int len = tem.size();
	for(int i=0,j=len-1;i<j;){
		if(tem[i]!=tem[j]) return false;
			i++;j--;
}
return true;
}};
```


[最长回文子串](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvn3ke/)
解法概述：回文串的长度为奇数或偶数，即形如a,aa或aba。
动态规划算法：定义二维数组bool dp(length)(length),dp(left)(right)\=\=true表示下标从left～right是回文字串
状态转移方程：dp(left)(right)=dp(left+1)(right-1) && s(left)\=\=s(right)
边界条件：
a) s(left)\!\=s(right),则跳过
b) s(left)\=\=s(right)：
0<=right-left<=2;dp(left)(right)\=\=true
right-left>2;dp(left)(right)\=dp(left+1)(right-1)
```c++
class Solution {
public:
string longestPalindrome(string s) {

int len = s.length();

bool dp[len][len];

memset(dp,0,sizeof(dp));
!!如果不初始化可能会因为内存有脏数据，而bool类型的数组只能容纳1或0，类型不匹配会报错

int start=0,maxLen=1;//保存最长回文串的起始位置和长度

for(int right=1;right<len;right++){

for(int left=0;left<right;left++){

	if(s[left]!=s[right])continue;
	if(left==right) dp[left][right]=true;//单个字母也是回文串
	else if(right-left <=2){
		dp[left][right]=true;
	}else{
		dp[left][right] = dp[left+1][right-1];	
	}
	if(dp[left][right]==1 && right-left+1>maxLen){
	//更新最长回文串长度并记录起始位置
	
		maxLen = right-left+1;
		start = left;
	}
}
}
return string(s.substr(start,maxLen));
}};
```

Manacher马拉车算法：
预处理：在每个字符串的两边及每个字符的两边插入一个字符串没有的特殊字符，使得无论字符串个数是奇数还是偶数都变成奇数。
回文半径：回文串中间字符到回文串最左边的长度叫回文半径。
目的：O(n)内求出以每个位置为回文中心的最长回文半径
操作：维护一个回文半径的数组r(c)表示字符c的最大回文半径是r(c)，再维护一个变量存储当前以某个字符作为回文中心的回文串是向右扩展最远的。

简单讲，利用回文镜像对称，用左边已有的结果加速判断右边；

形如：
```
第一种情况：i’的回文半径没有超过xy，则r[i]=r[i']
x---(-i'-)-c-(-i-)----y

第二种情况:r[i']超过xy边界，此时要截断r[i'],即r[i]=c+r[c]-i,即(y的坐标-i的坐标)，即iy段
(--x-i'---)-c--(--i--y)--
之后再暴力向两侧扩展r[i]

所以：
r[i] = min(c+r[c]-i,r[i']),后者即:r[i'] = r[2*c-i]
```

解析详情：
[manacher算法](https://blog.csdn.net/qq_40342400/article/details/124302078)
[参考2](https://zhuanlan.zhihu.com/p/611163818)
[manacher算法模板](https://www.luogu.com.cn/problem/P3805)
```c++
string getNewStr(string str){
    string temp="$#";
    for(char c:str){
        temp+=c;
        temp+="#";
    }
    temp+="^";
    //首尾各自加入不同特殊字符$^,充当哨兵角色，使得不用特殊判断边界
    return temp;
}//返回加工过的string字符串，用#分割字符，将字符串的长度变为固定的奇数

int main(){
    string str;cin>>str;
    str = getNewStr(str);
    int n = (int)str.size();
    vector<int>r(n,0);//回文半径，长度不包含当前字符
    auto f = [&](){
        int c = 0;//记录当前的最大回文半径的回文中心
        //没有使用mx记录最大的回文半径，因为r[c]+c就是最大的回文中心的回文半径
        for(int i=1;i<n-1;i++){
            //第一个字符是$无需判断，最后一个^无需判断
            if(c+r[c]>i)
                //如果当前的i在最大回文中心的回文半径之内
                r[i] = min(r[2*c-i],c+r[c]-i);//2种情况，一种是完全在范围内，一种是被截断，取二者中的最小值
            while(str[i-r[i]-1]==str[i+r[i]+1])
                r[i]++;
            if(i+r[i]>c+r[c])
                c = i;
            //如果有更大回文半径的回文中心，则更换
        }
    };
    f();
    cout<<*max_element(r.begin(), r.end());
    return 0;
}
```

[递增的三元子序列](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvvuqg/)
解法概述：注意为递增子序列，故三元组里的元素不一定连续。使用两个变量，一个记录扫描过的最小值，一个记录第二小值，更新情况如下：
a)当前数字小于最小值，则更新最小值
b)当前数字大于最小值，小于第二小值，则更新第二小值
c)如果当前数字大于第二小值，则找到递增三元子序列，返回true
```c++
class Solution {

public:

bool increasingTriplet(vector<int>& nums) {

int n = nums.size();

int small = 0x7fffffff,mid=0x7fffffff;//见c++差缺补漏数字特征部分

for(int i=0;i<n;i++){

if(nums[i]<=small){

small = nums[i];

}else if(nums[i]<=mid){

mid = nums[i];

}else{
return true;
}}

return false;
}};
```

[两数相加](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvw73v/)
解法概述：逐位模拟链表相加，用一个变量记录低位向高位的进位，注意最后还要再检测一次进位位。适当复习如何创建一个链表。
```c++
class Solution {

public:

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {

bool carry=false;//用于记录进位位
vector<ListNode*> n;

while(l1!=NULL && l2!=NULL){
int count = l1->val+l2->val+carry;
carry = count-10>0?1:0;
ListNode* node = new ListNode(count-10>0?count-10:count);
n.push_back(node);
l1 = l1->next;l2 = l2->next;
}

while(l1!=NULL){
int count = l1->val+carry;
carry = count-10>0?1:0;
ListNode* node = new ListNode(count-10>0?count-10:count);
n.push_back(node);
l1 = l1->next;
}

while(l2!=NULL){
int count = l2->val+carry;
carry = count-10>=0?1:0;
ListNode* node = new ListNode(count-10>=0?count-10:count);
n.push_back(node);
l2 = l2->next;
}

if(carry){
ListNode* node = new ListNode(1);
n.push_back(node);
}
!!这一步容易遗忘，最后依旧要检查是否有向高位的进位，如'999+999',最后还有一次进位要考虑

ListNode* p = NULL;
ListNode* head = NULL;
int s = n.size();
for(int i=0;i<s;i++){
ListNode* current = n[i];
if(i==0){
head = current;
p = head;
}

else{
p->next = current;
p = current;
}}
return head;
}};
```

[奇偶链表](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvdwtj/)
解法概述：使用双指针，一个指针用于连接奇数索引，另一个用于连接偶数索引，再设置两个指针分别指向奇数索引开头和偶数索引开头的位置，分别用于：奇数索引开头作为head返回，偶数节点开头用于连接两个链表。再设置一个暂存节点维护节点更新时的状态防止断链。
```c++
class Solution {
public:
ListNode* oddEvenList(ListNode* head) {
if(head==NULL) return head;
!! 千万注意边界条件
ListNode *odd = head, *even = head->next;
//odd存储的其实可以看作奇数链表的尾节点，even存储的其实可以看作偶数链表的尾节点。

ListNode* evenHead = even,*oddHead = odd;

ListNode* tmp = NULL;
//存储链表分离中的偶节点防止断链。

ListNode* preodd = odd;
//preodd存储的是奇数链表的前一个节点，用于最后连接奇偶索引链表。

while(1){
if(even==NULL){
preodd = odd;
break;
}else if(odd==NULL){
even->next = NULL;
break;
}
tmp = odd->next;
odd->next = even->next;
preodd = odd;

if(even!=NULL)
	odd = even->next;

if(odd!=NULL)
	even = odd->next;

tmp->next = even;
}
preodd->next = evenHead;
return oddHead;

}};
```

[相交链表](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xv02ut/)
解法概述：
1.朴素解法：先将一条链表放入集合set中，再遍历另一条链表，如果另一条链表中的节点在集合中存在，则这个节点就是交叉节点，否则返回NULL。
2.统计2个链表的长度，让长链表先遍历直到两个链表剩余长度相同，此时再同步检查看是否有相同节点即为交叉节点。
3.双指针：ab两个链表分别从各自头节点出发，并同时移动，此时会出现：
a):链表长度一致，则同时抵达终点
b):链表长度不同，则此时先后到达终点的两个指针不停止，从对方的链表再次出发，之后应同时停止与终点。----设a长度为m，b长度为n，则a走过m+n和b走过m+n，长度一致
双指针移动的时候不停检查节点是否重叠。时间复杂度为O(m+n)，空间复杂度为O(1).
```c++
class Solution {
public:
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {

ListNode *pa=headA,*pb=headB;
while(1){
if(pa==pb) return pa;
!! 注意要先检查是否直接相等，再判断是否是同时到达终点的情况

if(pa->next ==NULL && pb->next ==NULL) break;
//同时抵达终点，说明没有相交的情况

if(pa->next!=NULL) pa=pa->next;
if(pb->next!=NULL) pb = pb->next;
//同时移动两个指针

if(pa->next==NULL && pb->next!=NULL)
pa = headB;
//让走完的指针走另一个链表

if(pb->next==NULL && pa->next!=NULL)
pb = headA;
//让走完的指针走另一个链表
}
return NULL;
}};
```

[二叉树的中序遍历](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xv7pir/)
解法概述：
```c++
递归
class Solution {
public:
vector<int> inorderTraversal(TreeNode* root) {
vector<int>ans;
if(root==NULL) return ans;
middle(ans,root);
return ans;
}

void middle(vector<int>& ans,TreeNode* root){
if(root->left!=NULL)
middle(ans,root->left);

ans.push_back(root->val);

if(root->right!=NULL)
middle(ans,root->right);
}};

非递归：!!必看
class Solution {
public:
vector<int> inorderTraversal(TreeNode* root) {
stack<TreeNode*> s;
vector<int> ans;
if(root==NULL) return ans;
TreeNode* current = root;
while(current!=NULL || !s.empty()){
	while(current!=NULL){
		s.push(current);
		current = current->left;
	}
	if(!s.empty()){
		TreeNode* t = s.top();s.pop();
		ans.push_back(t->val);
		current = t->right;
	}
}

return ans;
}};
```

[二叉树的锯齿形层次遍历](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvle7s/)
解法概述：
锯齿形遍历：先从左往右，下一层再从右往左，每层交替进行。
即二叉树的层序遍历，用一个变量记录从左往右还是从右往左。
```c++
class Solution {
public:
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
vector<vector<int> > ans;
if(root==NULL) return ans;
queue<TreeNode*> q;
bool flag = 0;//0从左到右，1从右到左

q.push(root);
while(!q.empty()){

int level = q.size();//每次开始q.size()就是这一层的节点个数

vector<int> tmp;
for(int i=0;i<level;i++){//每层有多少个节点

TreeNode* top = q.front();q.pop();
tmp.push_back(top->val);
if(top->left!=NULL){
q.push(top->left);
}

if(top->right!=NULL){
q.push(top->right);
}
}

if(flag==0){
ans.push_back(tmp);flag = 1;
}else{
reverse(tmp.begin(),tmp.end());
ans.push_back(tmp);flag = 0;//交叉反转每层节点次序

}}
return ans;
}};
```

[从前序与中序遍历序列构造二叉树](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvix0d/)
解法概述：前序遍历找根节点，找到root后去中序遍历中划分左右子树集合，直到遍历完preorder列表。
注意：最好不要把变量定义在Solution类外面。
```c++
class Solution {
public:

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
int root = 0;//注意不要写在类外
return myBuild(0,preorder.size()-1,preorder,inorder,root);
}

TreeNode* myBuild(int left,int right,vector<int>& preorder, vector<int>& inorder,int& root){

if(left>right) return NULL;
TreeNode* rootNode = new TreeNode(preorder[root++]);
int i = left,n = right;
while(i<=n){
	if(inorder[i]==rootNode->val)
		break;
	i++;
}

rootNode->left = myBuild(left,i-1,preorder,inorder);
rootNode->right = myBuild(i+1,right,preorder,inorder);
return rootNode;
}};
```

[选择矩阵中单元格的最大得分](https://leetcode.cn/problems/select-cells-in-grid-with-maximum-score/)
解法概述：

状态压缩dp：[状压dp](https://blog.csdn.net/qq_57150526/article/details/128460546)
1.通常用01表示各个点的状态，如果有3状态也可以用三进制；
2.状态压缩的目的一方面缩小数据存储的空间，一方面在状态对比和状态整体处理时能提高效率；
3.状态数据中的单元个数不能太大，int表示状态时，状态个数不超过32

例子：
如4张卡，取0,1,3号，则为:1011(从右往左数);取2号则0100.
```c++
如果使用递归回溯算法：522/543，超时
struct node{
int x,y;
int val;
node(int xx,int yy):x(xx),y(yy){}
node(){}
bool operator<(const node& a)const{
return a.val < val;
}
}nod[15][15];

class Solution {// 522/543
public:
bool isSelected[15];
bool ismarked[150];
int maxValue = -1;
int maxScore(vector<vector<int>>& grid) {
int m = grid.size(),n = grid[0].size();
memset(isSelected,0,sizeof(isSelected));
memset(ismarked,0,sizeof(ismarked));

for(int i=0;i<m;i++){
for(int j=0;j<n;j++){
nod[i][j].x = i;
nod[i][j].y = j;
nod[i][j].val = grid[i][j];
}}
getScore(grid,0,0,0);
return maxValue;
}

void getScore(vector<vector<int> >& grid,int row,int tmpsum,bool flag){

if(flag) return;
if(row==grid.size() && tmpsum<=maxValue){
flag = true;
return;
}

if(row>=grid.size()){
if(maxValue<tmpsum)
maxValue = tmpsum;
return;
}

if(isSelected[row]==false){
isSelected[row] = true;
int n = grid[0].size();
sort(nod[row],nod[row]+n);//对某一行从大到小排列
bool fflag = false;//这一行是否有选择元素
for(int col=0;col<n;col++){
if(ismarked[nod[row][col].val]==false){
fflag = true;
ismarked[nod[row][col].val] = true;
getScore(grid,row+1,tmpsum+nod[row][col].val,flag);
ismarked[nod[row][col].val] = false;

}}

isSelected[row] = false;
if(fflag==false){
getScore(grid,row+1,tmpsum,flag);
}
}else
getScore(grid,row+1,tmpsum,flag);
}};
```

```c++
使用状态压缩dp

```


[填充每个节点的下一个右侧节点指针](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvijdh/)
解法概述：由于内存限制使用常量级额外空间，显然不可以使用层序遍历。
将每一行看作一个链表，使用一个冗余dummy节点，pre先指向这个作为横向链表头节点的节点，然后如从第二层开始，将第三层串起来。具体为：
pre = dummy;//初始状态
pre->next = cur->left;
pre = pre->next;
pre->next = cur->right;
pre = pre->next;
cur = cur->next;//继续横向去串同层的下一个节点。
重复上述操作可以将第三层节点依次串起，还要注意到是完美二叉树。叶子全在同一层节点,且有左节点就有右节点。
cur = dummy.next()
更新cur为下一层的第一个节点，然后继续将下下一层串起来。横向之间通过cur = cur->next进行遍历。
```c++
层序遍历法：
class Solution {
public:
Node* connect(Node* root) {

if(root==NULL) return NULL;

queue<Node*> q;
q.push(root);

while(!q.empty()){
int level = q.size();//这一层的个数
Node*pre = NULL;
for(int i=0;i<level;i++){

	Node* current = q.front();
	q.pop();
	if(pre!=NULL) pre->next = current;
	pre = current;
	if(current->left!=NULL){
		q.push(current->left);
	}
	if(current->right!=NULL){
		q.push(current->right);
	}
}}
return root;
}};
```

```c++
class Solution {
public:

Node* connect(Node* root) {

if(root==NULL) return NULL;
Node* pre,*cur = root;//cur其实遍历的是上一层节点

while(cur!=NULL){
	//确保cur最终可以指向完美二叉树的最后一层

	Node* dummy = new Node();
	pre = dummy;//pre实现将下一层串起来
	
	while(cur!=NULL && cur->left!=NULL){
	//cur!=NULL确保同一层还有下一个节点，且cur->left!=NULL保证这一层的下层还有节点可以横向串接
		pre->next = cur->left;
		pre = pre->next;
		pre->next = cur->right;
		pre = pre->next;
		cur = cur->next;
		//串完一个节点的左右节点后向同一层的下一个节点移动
	}
	cur = dummy->next;//一层串完则向下一层第一个开始遍历
}
return root;
}};
```

[填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)
解法概述：此题与上一题进行区分，本题的树形结构并非完美二叉树。
```c++
class Solution {
public:

Node* connect(Node* root) {
if(root==NULL) return NULL;

Node* cur = root;

while(cur!=NULL){

	Node* dummy = new Node();
	Node* pre = dummy;
	
	while(cur!=NULL){
		
		if(cur!=NULL && cur->left==NULL && cur->right==NULL){
			cur = cur->next;
		}
		//由于不是完美二叉树，所以可能出现中间某个节点左右子树为NULL的情况，跳过这个节点
		
		if(cur==NULL) break;//如果这一层节点遍历完了，就结束。
		
		if(cur->left!=NULL){
			pre->next = cur->left;
			pre = pre->next;
		}//不是完美二叉树，所以判断是否有左子树
		
		if(cur->right!=NULL){
			pre->next = cur->right;
			pre = pre->next;
		}//不是完美二叉树，所以判断是否有右子树
		cur = cur->next;//横向遍历节点
	}
	cur = dummy->next;//纵向走下一层的第一个节点
}
return root;
}};
```

[二叉搜索树中第k小的元素](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvuyv3/)
解法概述：
思路1:使用大顶堆找第k小的数，如果当前堆的大小小于k，则加入堆中；如果当前堆已经到达k，则：如果来的数比堆顶元素大，则直接丢弃；否则pop出堆顶元素并加入新到元素；
本题因为采用从大到小的遍历方式，所以可以直接弹出堆顶元素并加入新到元素；

思路2:统计节点个数：
1.左子节点的个数大于等于k，则要找的元素在左子节点
2.左子节点的数+1等于k，则当前节点就是要找的节点
3.否则去右子节点中找
```c++
思路1实现：
class Solution {
public:

bool cmp(int a,int b){
return a>b;
}

int kthSmallest(TreeNode* root, int k) {

priority_queue<int>q;//大顶堆实现第k小
traval(root,q,k);
return q.top();
}

void traval(TreeNode* root,priority_queue<int>& q,int k){

if(root->right!=NULL){
	traval(root->right,q,k);
}

if(q.size()<k)
	q.push(root->val);

else{
//因为遍历时采用从大到小的方式，所以直接弹出堆顶并加入新元素即可。
	q.pop();
	q.push(root->val);
}

if(root->left!=NULL)
	traval(root->left,q,k);
return;
}};
```
```c++
思路2实现：
class Solution {

public:

int kthSmallest(TreeNode* root, int k) {

int c = countNode(root->left);//计算左子节点的节点数

if(c>=k){//注意取等，k==c，则第k小的节点也在左子节点
	return kthSmallest(root->left,k);//直接在左子节点找
}else if(c+1==k){
	return root->val;//就是当前节点
}else{
	return kthSmallest(root->right,k-1-c);//去右子节点找k-1-c小的节点
}}

int countNode(TreeNode* root){
if(root==NULL){
//注意不要写成if(root->left==NULL && root->right==NULL)，因为每个父节点不一定都有两个子节点，所以再往下递归可能出现root==NULL的情况导致访问NULL
	return 0;
}
int left = countNode(root->left);
int right = countNode(root->right);
return left+right+1;
}};
```
[岛屿数量](https://leetcode.cn/leetbook/read/top-interview-questions-medium/xvtsnm/)
解法概述：使用dfs，从有1的点开始出发，途中找到一个1就将其变为0，然后继续探索；每次启动dfs一定能找到一片相邻的1（岛屿），启动dfs的次数就是岛屿的数量
```c++
class Solution {
public:
int numIslands(vector<vector<char>>& grid) {

int m = grid.size(),n = grid[0].size();
int ans = 0;

for(int i=0;i<m;i++)
for(int j=0;j<n;j++){
	if(grid[i][j]=='1'){//从每个岛屿开始遍历
	dfs(grid,i,j);
	ans++;
}}
return ans;
}

void dfs(vector<vector<char> >& grid,int i,int j){

int m = grid.size(),n = grid[0].size();

if(i<0 || i>=m || j<0 || j>=n) return;
if(grid[i][j]=='0') return;

if(grid[i][j]=='1'){
	grid[i][j] = '0';

// int direction[4][2] ={1,0,-1,0,0,1,0,-1};
// for(int k=0;k<4;k++){
// int new_x = i+direction[k][0];
// int new_y = j+direction[k][1];
// dfs(grid,new_x,new_y);
// }
上下两种方法均可以，均从某一个点的四个方向开始探索
	dfs(grid, i+1, j);
	dfs(grid,i-1,j);
	dfs(grid,i,j+1);
	dfs(grid,i,j-1);
	}}
};
```


## leetcode周赛
### 滑动窗口
双指针：满足单调性才可以使用，如子数组的和不断变小，while从满足要求变到不满足要求；右指针不断向后遍历，直到找到满足条件的子数组，然后收缩左指针，遍历所有符合条件的子数组(有时数组的开头也可以是left左侧的部分，见‘2962’)，直到left到达一个位置使得子数组不再符合条件，此时继续遍历右指针并重复上述过程；
[209.长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/description/)
```c++
//向右扩展遍历右端点，缩小左端点
//时间复杂度：O(n)
//计算ans是，判断l==r时，ans的个数应该是多少来判断是否+1；
//枚举右端点
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int len = nums.size();
        int ans = 0x7fffffff;
        int left=0,right=0;
        int tmpsum = 0;
        while(right<len && left<=right){
            while(right<len){
                if(tmpsum+nums[right]>=target){
                    tmpsum+=nums[right];
                    right++;
                    ans = min(ans,right-left);
                    break;
                }
                else{
                    tmpsum+=nums[right];
                    right++;
                }
            }
            while(1){//左端点范围不必判断
                if(tmpsum-nums[left]>=target){
                    tmpsum-=nums[left];
                    left++;
                    ans = min(ans,right-left);
                }
                else{
                    tmpsum-=nums[left];
                    left++;
                    break;
                }
            }
        }
        if(ans==0x7fffffff) return 0;
        else
        return ans;
    }
};
```
补充：前缀和+二分搜索做法：
$sums[i]$表示$nums[0,i-1]$求和，i从1到n遍历每个子数组的开始下标，时间复杂度为O(nlogn);
1.因为原数组中元素均为正数，所以sums前缀数组元素递增有序；
2.求$sums[k]-sum[j]>=target$,则k～j是一段满足条件的子数组；
3.继而求$sums[j]+target<=sums[k]$，令$s = sums[j]+target$
4.对sums数组进行二分查找，即可找到满足条件的k，即找：s<=$sums[k]$
5.区间长度即为：min(k-j),即所有求得满足条件的子数组的最小值；
```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = (int)nums.size();
        int ans = 0x7fffffff;
        vector<int> sums(n+1,0);//sums[i]为nums[0:i-1]求和
        sums[0] = 0;
        sums[1] = nums[0];
        for(int i=1;i<=n;i++)
            sums[i] = sums[i-1]+nums[i-1];
            //[0~i-1]=[0~i-2]+[i-1]
        
        for(int i=1;i<=n;i++){
            int s = target+sums[i-1];
            auto bound = lower_bound(sums.begin(), sums.end(), s);
            //bound指向大于等于(target+某一段和)的元素位置
            //对于每个开始下标i，用二分查找找到长度最小的子数组
            //bound对应解说中的k
            //子数组的长度：k-j=(bound-begin()) - (j-1)-减1是调整下标
            if(bound!=sums.end()){
                ans = min(ans,static_cast<int>(bound-sums.begin())-(i-1));
            }
        }
        return ans==0x7fffffff?0:ans;
    }
};

非调用函数版本：
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = (int)nums.size();
        int ans = 0x7fffffff;
        vector<int> sums(n+1,0);//sums[i]为nums[0:i-1]求和
        sums[0] = 0;
        sums[1] = nums[0];
        for(int i=1;i<=n;i++)
            sums[i] = sums[i-1]+nums[i-1];//[0~i-1]=[0~i-2]+[i-1]
        
        for(int i=1;i<=n;i++){
            int s = target+sums[i-1];
            int l=1,r=n;//sums的范围是1～n
            int mid = -1;
            while(l<=r){
                mid = (l+r)>>1;
                if(sums[mid]>=s){
                    r = mid-1;
                }else{
                    l = mid+1;
                }
            }
            if(l<=n && sums[l]>=s)//注意判断l不要越界
                ans = min(ans,l+1-i);
        }
        return ans==0x7fffffff?0:ans;
    }
};

注意二分查找的写法：
如数组 v={2,5,6,8,12,15},查找第一个大于等于7的元素的位置,l=0,r=5
mid = 2,v[mid] = 6,更新l=mid+1=3,r=5
mid = 4,v[mid] = 12,更新r=mid-1=3,l=3
mid = 3,v[mid] = 8,更新r=mid-1=2,l=3
结束，此时检查l对应的元素v[l]=8，是第一个大于等于7的元素，更新ans

若查找第一个大于等于20的元素的位置,l=0,r=5
mid = 2,v[mid]=6,更新l=mid+1=3,r=5
mid = 4,v[mid]=12,更新l=mid+1=5,r=5
mid = 5,v[mid]=15,更新l=mid+1=6,r=5
结束，此时由于l超过范围，所以不更新ans,说明数组中不存在这样满足条件的位置
```

[713.乘积小于k的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/description/)
```c++
//类似上题，如果累积小于k，则移动right，否则缩小left；
//单调性：while(mul>=k)从满足条件到不满足条件
//特判断k==0和k==1无法满足"子数组"，"乘积严格小于k"的条件
//推导公式，[l,r]中累积若小于k，则以区间右端点为r的子数组，[l+1,r]也小于k,.,[r,r]也小于k,即r-l+1
//枚举右端点
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if(k==0||k==1)return 0;
        int len = nums.size();
        int left = 0,right = 0;
        int mul = 1;
        int ans = 0;
        while(right<len){
            mul *= nums[right];
            while(mul>=k){
                mul /= nums[left];
                left++;
            }
            ans += right-left+1;
            right++;
        }
        return ans;
    }
};
```

[3.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
```c++
//双指针，用right遍历字符，用字典存储每个字符出现的次数；
//如果遇到一个字符出现次数大于1，则移动左指针，且使每个左指针对应字符出现次数-1，直到右指针所对应的字符出现次数等于1，说明此时[left,right]中没有重复的字符，且已经把left和right所夹区间之外的字符出现次数归零(即维护一个最长子区间
//时间复杂度：O(n)
//空间复杂度：O(set(s))
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = s.size();
        map<char,int> chk;
        int ans = 0;
        int left=0,right=0;
        while(right<len){
            chk[s[right]]++;
            while(chk[s[right]]>1){
                chk[s[left]]--;
                left++;
            }
            ans = max(ans,right-left+1);
            right++;
        }
        return ans ;
    }
};
```
[2962.统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/description/)
```c++
/*
思路：双指针滑动窗口：设置left和right两个指针，right遍历数组的每个元素，同时设mx为数组中的最大值,cntMax记录等于最大值的个数。
如果nums[right]==mx,cntMax++;
如果cntMax==k,右移left指针直到窗口内的mx出现次数小于k为止;
对于右端点right且左端点小于left的子数组，mx的出现次数至少为k;
即此时，以left之前的元素为起始元素的子数组均为满足条件的子数组；ans+=left;

理解：先用right找到一个窗口，使其中包含k个mx值；然后移动left，找满足条件的子数组；left叠加了之前的子数组的结果；left=left_pre+当前[l,r]满足条件数组
如有数组：1 2 3 100 4 100 100 3 100,k=2,mx=100
right第一次停在:nums[5],此时遍历left可以找到:
1 2 3 100 4 100
2 3 100 4 100
3 100 4 100
100 4 100--此时left停在nums[4],共4个
right第二次停在:nums[6],此时可以找到:
1 2 3 100 4 100 100
2 3 100 4 100 100
3 100 4 100 100
100 4 100 100
4 100 100
100 100--此时left停在nums[6]，共6个
right第三次停在:nums[7],此时可以找到:
1 2 3 100 4 100 100 3
...
100 100 3---此时left停在nums[6]
right第四次停在:nums[8],此时可以找到:
1 2 3 100 4 100 100 3 100
...
100 3 100--此时left停在nums[7]
此时算法结束；

数组如下,k=2:
       1 2 3 max 4 max max 3 max
	   l            r
移动l:            l  r
此时，l=4,以left左边元素为开头到right的子数组刚好4个,均包含k个mx值；
	   1 2 3 max 4 max max 3 max
	             l      r
移动l:                  l,r
此时，l=6,即包含k个mx的子数组个数为6个；
	   1 2 3 max 4 max max 3 max
	                    l  r
此时，l=6,即包含k个mx的子数组个数为6个；--容易遗漏
	   1 2 3 max 4 max max 3 max
	                    l     r
移动l:                      l  r
此时，l=7,即包含k个mx的子数组有7个；
所以最终ans=4+6+6+7=23
*/
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        int len = nums.size();
        int left = 0;
        long long cntMax = 0;
        int m = -1;
        long long ans = 0;
        for(int i=0;i<len;i++)
            m = max(nums[i],m);
        for(int right=0;right<len;right++){
            if(nums[right]==m){
                cntMax++;
            }
            while(cntMax==k){
            //不满足条件才退出循环，所以直到退出循环的前一刻是满足要求的
            //所以退出循环的前一刻之前的位置都是满足题目要求的
                if(nums[left]==m)
                    cntMax--;
                left++;
            }
            ans+=left;
        }
        return ans;
    }
};
```

[3305.元音辅音字符串计数I](https://leetcode.cn/problems/count-of-substrings-containing-every-vowel-and-k-consonants-i/description/)
[3306.元音辅音字符串计数II](https://leetcode.cn/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/description/)
```c++
/*
1.恰好包含k个，可以转化为：>=k,>=k+1,两个数量相减即为==k;
2.“至少”问题类比2962，当right指针遍历到一个符合题目要求的位置，此时收缩左指针，从满足条件->不满足条件，此时[0,left-1]均为满足条件的子数组开头；
*/
class Solution {
public:
    long long countOfSubstrings(string word, int k) {
        return f(word,k)-f(word,k+1);
        //恰好问题转换为两个不等式值的差，即num(>=k)-num(>=k+1)
    }
    long long f(string word,int k){
        map<char,long long> vowel;//map统计aeiou的数量
        int cnt=0;//变量计数辅音
        int left = 0;
        long long ans = 0;
        int len = word.size();
        for(int right=0;right<len;right++){
            if(word[right]=='a'||word[right]=='e'||word[right]=='i'||word[right]=='o'||word[right]=='u')
                vowel[word[right]]++;//如果是aeiou，则map[key]++
            else cnt++;//辅音个数+1
            while(vowel.size()==5 && cnt>=k){
                if(word[left]=='a'||word[left]=='e'||word[left]=='i'||word[left]=='o'||word[left]=='u'){
                    vowel[word[left]]--;
                    if(vowel[word[left]]==0)
                        vowel.erase(word[left]);
                }
                else cnt--;
                left++;
            }
            //当不满足上述while时才退出，即[0,left-1]位置均满足题目要求，所以ans+=left即加入满足aeiou至少出现一次，且包含至少k个辅音字母的子字符串数
            ans+=left;
        }//遍历右指针，收缩左指针，类比2962
        return ans;
    }
};
```

### 第418场
[3309.连接二进制表示可形成的最大数值](https://leetcode.cn/problems/maximum-possible-number-by-binary-concatenation/description/)
思路：
法1:因为$nums.size()==3$,所以可以直接手动计算每一种连接得到元素的值取最大即可；
法2：把nums排序，对相比较的两个元素a、b,如果其二进制表示进行或运算后a+b>b+a,那么a就应该排在b左边；--如100+11=10011；11+100=11100
```c++
class Solution {
public:
    int maxGoodNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end(),[](int a,int b){
            int high_a = (int)log2(a)+1;
            //计算a最左边的1的位数+1，为后续拼上b多腾一个位置
            //使用log2计算，可以找到最高位的1，且无前导0
            int high_b = (int)log2(b)+1;
            //b同理
            return (a<<high_b | b)>(b<<high_a |a);
            //a+b>b+a,其中+表示二进制或
        });
        int ans = 0;
        for(auto i:nums){
            ans = ans<<((int)log2(i)+1)|i;
            //总结：通过左移和或操作进行循环拼接
        }
        return ans;
    }
};
```
类似题：
[179.最大数](https://leetcode.cn/problems/largest-number/description/)
思路：
因为返回的是字符串，所以可以尝试直接拼接数字，然后将其转成字符串，进而变为对字符串字典序的比较；
```c++
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end(),[](int a,int b){
            string s1 = to_string(a);
            string s2 = to_string(b);
            return s1+s2>s2+s1;//a+b>b+a
        });
        if(nums[0]==0)
            return "0";
        //特殊判断,如[0,0]的情况
        string ans;
        for(auto i:nums)
            ans+=to_string(i);
        return ans;
    }
};
```

[3310.移除可疑的方法](https://leetcode.cn/problems/remove-methods-from-project/description/)
思路：

```c++

```


### 第419场
[3318.计算子数组的x-sum I](https://leetcode.cn/problems/find-x-sum-of-all-k-long-subarrays-i/description/)
思路：
模拟计算过程返回结果即可，使用优先队列进行排序，按照出现次数降序排序，相同则按照元素大小降序排序；
```c++
class Solution {
public:
    vector<int> findXSum(vector<int>& nums, int k, int x) {
        int n = nums.size();
        vector<int>ans;
        for(int l = 0;l<=n-k;l++){
            int tmp = cal(l,l+k,nums,x);
            ans.push_back(tmp);
        }
        return ans;
    }
    int cal(int l,int r,vector<int> arr,int x){//左闭右开
        unordered_map<int, int>cnt;
        for(int i=l;i<r;i++)
            cnt[arr[i]]++;
        if(cnt.size()<x){
            return accumulate(arr.begin()+l, arr.begin()+r, 0);
        }
        
        priority_queue<pair<int,int>,vector<pair<int,int> > >q;//key:次数 value:大小
        for(auto i:cnt)
            q.push({i.second,i.first});
        int tmpsum = 0;
        for(int i=0;i<x;i++){
            pair<int,int> t = q.top();
            tmpsum += t.second*t.first;
            q.pop();
        }
        return tmpsum;
    }
};
```

[3319.第K大的完美二叉子树的大小](https://leetcode.cn/problems/k-th-largest-perfect-subtree-size-in-binary-tree/description/)
思路：
首先读清题目，第k大的完美二叉子树：第k大判断的是二叉子树的结点个数按照降序排列排第k个；利用递归，如果左右子树高度一致，则当前root就是完美二叉子树，高度为left_high+right_high+1,否则就不是，标记为-1;
```c++
class Solution {
public:
   int kthLargestPerfectSubtree(TreeNode* root, int k) {
       vector<int>ans;
       getNum(root, ans);
       int c = 0;
       int len = ans.size();
       for(int i=0;i<ans.size();i++){
           if(ans[i]==-1)
               c++;
       }
       if(len-c<k) return -1;
       
       sort(ans.begin(),ans.end(),[&](int a,int b){
           return a>b;
       });
       return ans[k-1];
   }
    int getNum(TreeNode* root,vector<int>& ans){
        if(!root) return 0;
        int lh = getNum(root->left,ans);
        int rh = getNum(root->right, ans);
        if(lh==rh){
            int curh = rh+lh+1;
            ans.push_back(curh);
            return curh;
        }
        return -1;
    }
    
};
```


### 第420场
[3324.出现在屏幕上的字符串序列](https://leetcode.cn/problems/find-the-sequence-of-strings-appeared-on-the-screen/description/)
思路：
用一个char数组，保存每一步需要变得字符，使用loc变量控制当前正在处理的字符的位置；使用string的构造函数直接将char数组变为string类；
```c++
class Solution {
 public:
     vector<string> stringSequence(string target) {
         vector<string>ans;
         int loc = 0;
         char tmp[410];
         for(char c:target){
             int n = c-'a';
             for(int i=0;i<=n;i++){
                 tmp[loc] = 'a'+i;
                 ans.push_back(string(tmp,tmp+loc+1));
             }
             loc++;
         }
         return ans;
     }
 };
```

[3325.字符至少出现K次的子字符串 I](https://leetcode.cn/problems/count-substrings-with-k-frequency-characters-i/)
思路：
关键字：子串，越长越合法(单调性)，滑动窗口
一个例子：
如:$abacb$且k=2：当r=2时，子串$aba$合法，则此时合法子串有：
$abac$,$abacb$,$aba$，即1个$aba$串加上后面字符的个数，即1+2=3； 
收缩左端点，即l=1，此时$ba$不合法，即再次扩大右端点r，直到r=4，此时：
$bacb$合法，即1个$bacb$加上后面字符个数0,即1;
最终，得到结果个数为:3+1=4;

另一种写法：
$dcabacb$，当l=2,r=4时，如果$aba$合法，则向左$caba$,$dcaba$均合法；所以现在目标就是找到这个l=2，l++，然后把l之前的所有cnt都减少，同时ans+=l；
```c++
class Solution {
 public:
     int numberOfSubstrings(string s, int k) {
         int n = (int)s.size();
         int ans = 0;
         for(int l=0;l<n;l++){
             unordered_map<char, int> cnt;
             int vc = 0;
             for(int r=l;r<n;r++){
                 cnt[s[r]]++;
                 if(cnt[s[r]]==k){
                     vc++;
                 }
                 else if(cnt[s[r]]==k+1)
                     vc--;
                 if(vc>0){
                     ans += (n-r);
                     break;
                 }
             }
         }
         return ans;
     }
 };

另一种写法：
class Solution {
public:
    int numberOfSubstrings(string s, int k) {
        int n = s.size();
        int ans = 0;
        unordered_map<char, int>cnt;
        int l = 0,r = 0;
        for(;r<n;r++){
            cnt[s[r]]++;
            while(cnt[s[r]]>=k){
                cnt[s[l]]--;
                l+=1;
            }
            ans+=l; 
        }
        return ans;
    }
};
```

[3326.使数组非递减的最少除法操作次数](https://leetcode.cn/problems/minimum-division-operations-to-make-array-non-decreasing/)
思路：
1.分析计算的结果:x=4,y=2；x=6,y=2；x=8,y=2；x=12,y=2；x=15,y=3；x=75,y=3,即将x变为其最小质因子；
2.变成质数后不可再变，每个数最多操作一次变为最小质因子；
3.因为把每个数变小，所以数组中的最后一个数不需要变；(如果把每个数变大，则数组中的第一个数不需要变)
```c++
class Solution {
 public:
     int minOperations(vector<int>& nums) {
         int n = nums.size();
         int ans = 0;
         for(int i=n-1;i>=1;i--){
             while(nums[i]<nums[i-1]){
                 int fact = getfactor(nums[i-1]);
                 if(fact==1)
                     return -1;
                 nums[i-1]/=fact;
                 ans++;
             }
         }
         return ans;
     }
     int getfactor(int n){
         int tmpmax = -1;
         for(int i=2;i*i<=n;i++){
             if(n%i==0){
                 tmpmax = max(tmpmax,n/i);
             }
         }
         return tmpmax==-1?1:tmpmax;
     }
 };

根据思路，写出第二种做法：
const int MX =1000000;
int LPF[MX+1];
//注意一定要用const定义MX并将数组开在堆上
auto f = [](){
    for(int i=2;i<=MX;i++){
	    //遍历数组元素的取值范围
        if(LPF[i]==0){
        //0为特殊值，表示没有标记过（也即最小质因子）
            for(int j=i;j<=MX;j+=i)
                if(LPF[j]==0)
	                //标记那些没被标记过的数
                    LPF[j] = i;
        }
    }
};
//预处理类似线性筛素数，从2开始将其倍数的最小质因子进行标记(如4，6，8..的最小质因子标记为2)
class Solution {
public:
    int minOperations(vector<int>& nums) {
        int n = (int)nums.size();
        int ans = 0;
        f();
        for(int i=n-2;i>=0;i--){
            if(nums[i]>nums[i+1]){
                nums[i] = LPF[nums[i]];
                ans++;
                if(nums[i]>nums[i+1])
                    return -1;
            }
        }
        return ans;
    }
};
```

[3327. 判断 DFS 字符串是否是回文串](https://leetcode.cn/problems/check-if-dfs-strings-are-palindromes/)
思路：
前验知识：
1.DFS时间戳；
2.Manacher算法；
3.子树的后序遍历的字符串是整棵树后序遍历字符串的子串；

细节：
时间戳：在DFS一棵树的时候，维护一个全局的时间戳clock，每访问一个新的结点，就将clock加一，同时记录进入结点x时的时间$in[x]$和离开这个结点时的时间戳$out[x]$.
时间戳性质：设有以x为根的树，y为x的孙子结点，则：必须先递归完以y为根的子树之后才能递归完以x为根的树；则：如果y是x的子孙结点，则区间$[in[y],out[y]]$被区间$[in[x],out[x]]$包围,反之亦然；因此可以通过$in[x]<in[y]\leq out[x]$判断x是否为y的祖先结点；


```c++
//未完成，只是构建了后序遍历
using namespace std;
string getNewStr(string str){
    string temp="$#";
    for(char c:str){
        temp+=c;
        temp+="#";
    }
    temp+="^";
    //首尾各自加入不同特殊字符$^,充当哨兵角色，使得不用特殊判断边界
    return temp;
}//返回加工过的string字符串，用#分割字符，将字符串的长度变为固定的奇数
//class Solution {
//public:
//    vector<bool> findAnswer(vector<int>& parent, string s) {
//
//    }
//};
struct treeNode{
    char c;
    treeNode* left;
    treeNode* right;
    treeNode(char t){
        c = t;
        left =NULL;
        right = NULL;
    }
    treeNode(){
        left = NULL;
        right = NULL;
    }
};

int main(){
    vector<int> parent{-1,0,0,1,1,2};
    string str = "aababa";
    int n = parent.size();
    vector<treeNode*> tree(2*n,NULL);
    for(int i=0;i<n;i++){
        if(parent[i]==-1){
            treeNode* root = new treeNode(str[i]);
            tree[0] = root;
        }else{
            treeNode* node = new treeNode(str[i]);
            if(tree[parent[i]]->left==NULL){
                tree[parent[i]]->left = node;
                tree[2*parent[i]+1] = tree[parent[i]]->left;
            }
            else if(tree[parent[i]]->right==NULL){
                tree[parent[i]]->right = node;
                tree[2*parent[i]+2] = tree[parent[i]]->right;
            }
        }
    }
    for(int i=0;i<n;i++){
        cout<<tree[i]->c<<" ";
    }//建树，并层序遍历
    cout<<endl;
    auto f = [&](auto&& f,int index)->void{
        if(index*2+1<n && tree[index*2+1]!=NULL)
            f(f,index*2+1);
        if(index*2+1<n && tree[index*2+2]!=NULL)
            f(f,index*2+2);
        cout<<tree[index]->c<<" ";
        return;
    };//后序遍历
    f(f,0);
    return 0;
}
```

### 第142场双周赛 
[Q1. 找到初始输入字符串 I](https://leetcode.cn/contest/biweekly-contest-142/problems/find-the-original-typed-string-i/)
思路：
```c++

```

[Q2. 修改后子树的大小](https://leetcode.cn/contest/biweekly-contest-142/problems/find-subtree-sizes-after-changes/)
思路：
```c++

```

### 第421场



## 动态规划
[动态规划-基础版](https://gleetcode.cn/studyplan/dynamic-programming/)

[70.爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
由于每次只能上1个台阶或2个台阶，所以1层楼梯只有一种爬法，2层楼有2种(1+1或2)，接下来的每一阶均可有前一级台阶或前两级台阶走到，所以转移方程为：$dp[i]=dp[i-1]+dp[i-2]$,其中$dp[i]$表示第i层楼梯有多少种走法（注意为方案数，即虽然2楼可以走2次1级阶梯，但只算做一种方案数，dp(2)=2,因为还可以一步走2级阶梯）；
```c++
class Solution {
public:
int climbStairs(int n) {
int dp[46];
memset(dp,0,sizeof(dp));
dp[1]=1;dp[2]=2;
for(int i=3;i<=n;i++)
	dp[i] = dp[i-1]+dp[i-2];
return dp[n];
}};
```

[509.斐波那契数列](https://leetcode.cn/problems/fibonacci-number/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
1.两个局部变量，轮流计算f(n)=f(n-1)+f(n-2);
2.动态规划：$dp[i]$表示第i个斐波那契数列，动态转移方程为 ：$dp[i]=dp[i-1]+dp[i-2]$
两个局部变量即可：
```c++
class Solution {
public:
int fib(int n) {
int a=0,b=1;
if(n<2){
	if(n==0) return a;
	else return b;
}

for(int i=2;i<=n;i+=2){
a = a+b;
b = a+b;
}
if(n%2==0)return a;
else return b;
}};
```
动态规划版：
```c++
class Solution {
public:
int fib(int n) {
int dp[31];
dp[0]=0;dp[1]=1;
for(int i=2;i<=n;i++)
	dp[i] = dp[i-1]+dp[i-2];
return dp[n];
}};
```

[1137.第N个泰波那契数](https://leetcode.cn/problems/n-th-tribonacci-number/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
同斐波那契数列一致，转移方程为：$dp[i]=dp[i-1]+dp[i-2]+dp[i-3]$
```c++
class Solution {
public:
int tribonacci(int n) {
int dp[38];
dp[0]=0;dp[1]=1;dp[2]=1;
for(int i=3;i<=n;i++)
	dp[i] = dp[i-1]+dp[i-2]+dp[i-3];
return dp[n];
}};
```

[746.使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
注意可以选择从下标为0或为1的位置开始；即计算cost数组的一个子序列，这些子序列的间隔为1或2，使得这些字序列的代价和最小；
动态规划：$dp[i]$定义为到楼梯i的位置时所花费的最小代价，因为最开始可以从楼梯下标为0或1的地方出发，所以要初始化这两个dp，且动态转移方程中:
$dp[i]$要从前1阶或前2阶楼梯转移过来；且最后由于要求到达楼梯顶，所以能登顶的楼梯是$dp[len-1]$向上跨一阶，或$dp[len-2]$向上跨两阶，最后需要返回这两个位置的最小dp值；
```c++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int len = cost.size();
        int dp[len];
        memset(dp,0x7fffffff,sizeof(dp));
        dp[0] = cost[0];
        dp[1] = min(cost[1],dp[0]+cost[1]);
        for(int i=2;i<len;i++){
            dp[i] = min(dp[i-1]+cost[i],dp[i-2]+cost[i]);
        }
        return min(dp[len-2],dp[len-1]);
    }
};
```

[354.俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/description/)
思路：
对二维数组(w,h)中的第一维w进行升序排序，后对第二维h进行降序排序，之后求h的最长递增子序列即可；---需要用二分来降低时间复杂度，否则$O(n^2)$超时

注意：
1.本题中自定义sort排序用的lambda表达式的写法；
2.如果维度升高到三维，需要借助树状数组；
```c++
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end(),[](const auto&a,const auto&b){
            if(a[0]==b[0])return a[1]>b[1];
            else return a[0]<b[0];
        });
        vector<int> dp;
        int len = envelopes.size();
        dp.resize(len);
        fill(dp.begin(),dp.end(),1);
        
        for(int i=1;i<len;i++){
            for(int j=i-1;j>=0;j--){
                if(envelopes[j][1]<envelopes[i][1])
                    dp[i] = max(dp[i],dp[j]+1);
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

[53.最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)
思路：
动态规划，定义$dp[i]$为以$nums[i]$结尾的最大子数组和；最后返回整个dp数组的最大值，即为给定数组的最大子数组和；$dp[i]$要么与前面相邻的子数组连接，形成更大的子数组；要么自己作为新的子数组开头；
```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int dp[len];
        for(int i=0;i<len;i++)dp[i] = nums[i];
        for(int i=1;i<len;i++){
            dp[i] = max(dp[i],dp[i-1]+nums[i]);
        }
        return *max_element(dp,dp+len);
    }
};
```

[198.打家劫舍](https://leetcode.cn/problems/house-robber/?envType=study-plan-v2&envId=dynamic-programming)
思路：
注意，如果两家相邻的房屋在同一晚被小偷闯入，系统会自动报警；
使用动态规划，注意条件不要从相邻的前一个位置开始即可，本质是最长递增子序列的变形；
```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        int dp[len];
        for(int i=0;i<len;i++)
            dp[i] = nums[i];
        
        for(int i=1;i<len;i++){
            for(int j=i-2;j>=0;j--){
                dp[i] = max(dp[i],dp[j]+nums[i]);
            }
        }
        return *max_element(dp, dp+len);
    }
};
```

[740.删除并获得点数](https://leetcode.cn/problems/delete-and-earn/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
注意，删除单个$nums[i]$后,必须删除的是所有等于$nums[i-1]$和$nums[i+1]$的数
，所以可以排序，顺序不影响判断；用map记录下每个$nums[i]$出现的次数，并且可对$nums$去重；
```c++
class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        if(nums.size()==1)return nums[0];
        set<int> st;//保存一个无重复，升序的原数组元素
        map<int,int> chk;//key是数字，value是对应出现的个数
        
        st.insert(0);//插入0避免边界讨论
        for_each(nums.begin(), nums.end(), [&](const int& key){
            chk[key]+=1;
            st.insert(key);
        });
        vector<int> arr(st.begin(),st.end());
        //升序无重复数组，第一个元素为0避免边界讨论
        int len = (int)arr.size();
        vector<int> dp(len,0);
        dp[0] = arr[0]*chk[arr[0]];//即dp[0] = 0
        dp[1] = arr[1]*chk[arr[1]];
        //第一个元素:dp[1]=arr[1]*chk[arr[1]]
        int last = arr[1];//上一个判断的arr[1]
        for(int i=2;i<len;i++){
            dp[i] =arr[i]*chk[arr[i]];//先计算只选当前元素得到的点数
            if(arr[i]-1==last){
                last = arr[i];
                dp[i] = max(dp[i-1],dp[i-2]+dp[i]);
                //前面补0的原因，此处可以直接i-2，不需要边界判断
                //dp[i]要么是只拿前一个元素得到的点数(即跳过当前元素)，要么是拿前前元素的点数与当前点数之和(即跳过前一个元素)
            }
            else{
                last = arr[i];
                dp[i] = dp[i-1]+dp[i];//如果不冲突，则拿前一个元素的点数加上当前元素的点数
            }
        }
        return dp[len-1];
    }
};
```


[62.不同路径](https://leetcode.cn/problems/unique-paths/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
1.组合排列：从左上角移动到右下角需要总共：m-1+n-1=m+n-2次，路径的总数就等于：从m+n-2次总路径中选出m-1次向下移动的次数，即$C_{m+n-2}^{m-1}$；
2.动态规划：$dp[i][j]$表示从起点到达(i,j)的路径数，且(i,j)只能从上方或左侧到达；所以状态转移方程为：$dp[i][j]=dp[i-1][j]+dp[i][j-1]$，当位于第一行或第一列，即$dp[0][j]$或$dp[i][0]$时，dp为1，即此时只能从起点走一条路(直下或直右)过来；
```c++
class Solution {
public:
    int uniquePaths(int m, int n) {//m行n列
        vector<vector<int> > dp(m,vector<int>(n,0));
        //dp[i][j]表示达到(i,j)位置的路径数
        //dp[i][j] = dp[i-1][j]+dp[i][j-1]
        //即每个(i,j)只能从左边或者上面转移过来，等于左边的路径数+上边的路径数
        for(int i=0;i<m;i++)
            dp[i][0] = 1;
        for(int j=0;j<n;j++)
            dp[0][j] =1;
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++)
                dp[i][j] = dp[i][j-1]+dp[i-1][j];
        }
        return dp[m-1][n-1];
    }
};
或
class Solution {
public:
    int uniquePaths(int m, int n) {//m行n列
        long long ans = 1;
        for(int x=n,y=1;y<m;x++,y++)
            ans = ans*x/y;//公式见思路1
            //不要写成ans*=x/y;当x=3,y=2时由于向下舍入，导致x/y=1
        return ans; 
    }
};
```

[64.最小路径和](https://leetcode.cn/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
动态规划，类似题62，都是(i,j)只能从上方或左侧移动过来，但是初始化步骤不同，将$dp[0][0]$初始化为$grid[0][0]$，且$dp[0][j]$和$dp[i][0]$即第一行第一列初始化为前一个dp累加当前grid值，后续状态转移方程同62题一致，不过要记得加上grid当前位置的值；
```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(),n = grid[0].size();
        vector<vector<int> > dp(m,vector<int>(n,0));
        dp[0][0] = grid[0][0];
        for(int i=1;i<m;i++)
            dp[i][0] =dp[i-1][0]+grid[i][0];
        for(int j=1;j<n;j++)
            dp[0][j] = dp[0][j-1]+grid[0][j];
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++)
                dp[i][j] = min(dp[i-1][j]+grid[i][j],dp[i][j-1]+grid[i][j]);
        }
        return dp[m-1][n-1];
    }
};
```

[63.不同路径II](https://leetcode.cn/problems/unique-paths-ii/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
注意一个点，有可能出现v{{0,0},{1,1},{0,0}}的情况，即从左上角没有一条路可以到右下角;也有可能(0,0)直接就是障碍物；$dp[i][j]$表示从起点到(i,j)的路径数；
讨论思路：
1.如果当前格子是障碍，即$grid[i][j]==1$,则$dp[i][j]==-1$表示不可达；
2.如果当前格子左侧、上方均不可达，即$dp[i-1][j]==dp[i][j-1]==-1$,则$dp[i][j]==-1$，当前格子也不可达；
3.$dp[i][j]$只能从上方或左侧可达的格子过来；
4.处理边界的时候，要时刻注意前一个格子是否可达；
5.特别判断(0,0)处即起点是否有障碍；
6.最后判断$dp[m-1][n-1]$是否可达,即是否为-1，如果是返回0，否则返回dp值
```c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(),n = obstacleGrid[0].size();
        vector<vector<int> > dp(m,vector<int>(n,0));
        //dp[i][j]==-1表示不可达，否则表示从左上角到(i,j)的路径数
        if(obstacleGrid[0][0]==1)
            return 0;
        dp[0][0] = 1;
        for(int i=1;i<m;i++){
            if(obstacleGrid[i][0]==1){
                dp[i][0] = -1;
                continue;
            }
            if(dp[i-1][0]!=-1){
                dp[i][0] = 1;
            }//如果前一个可达
            else dp[i][0] = -1;
        }
        for(int j=1;j<n;j++){
            if(obstacleGrid[0][j]==1){
                dp[0][j] = -1;
                continue;
            }
            if(dp[0][j-1]!=-1)
                dp[0][j] = 1;
            else dp[0][j] = -1;
        }
        
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(obstacleGrid[i][j]==0){
                    if(dp[i-1][j]!=-1)
                        dp[i][j] += dp[i-1][j];
                    if(dp[i][j-1]!=-1)
                        dp[i][j] += dp[i][j-1];
                    if(dp[i-1][j]==-1 && dp[i][j-1]==-1)
                        dp[i][j] = -1;
                }//如果当前格子没有障碍
                else{
                    dp[i][j] = -1;
                }
                
            }
        }
        return dp[m-1][n-1]==-1?0:dp[m-1][n-1]; 
    }
};
```


[120.三角形最小路径和](https://leetcode.cn/problems/triangle/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
湖南大学经典动态规划题，从最后一层逐层向上计算，上一层的(i,j)由这一层的(i+1,j)和(i+1,j+1)过渡，设$dp[i][j]$表示到(i,j)处的最短最小路径和，状态转移方程为:$dp[i][j]=min(dp[i+1][j]+nums[i][j],dp[i+1][j+1]+nums[i][j])$
```c++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size(),n = triangle[m-1].size();
        vector<vector<int> > dp(m,vector<int>(n,0));
        for(int j=0;j<n;j++)
            dp[m-1][j] = triangle[m-1][j];//从最后一层开始
        for(int i=m-1;i>=1;i--){//从最后一层开始,到倒数第二层
            for(int j=0;j<i;j++){
                dp[i-1][j] = min(dp[i][j]+triangle[i-1][j],dp[i][j+1]+triangle[i-1][j]);
            }
        }
        return dp[0][0];
    }
};
```

[931.下降路径最小和](https://leetcode.cn/problems/minimum-falling-path-sum/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
本质还是题120的变形，不过从三角形，即上方2个来的方向变成矩阵，即上方3个来的方向，$dp[i][j]$表示达到(i,j)位置的路径和，最后在矩阵的最后一行找最小值即可；在dp的每行首尾插入一个大数可以有效避免如每行的第0列/每行的最后一列的边界判断，大数不会影响最终最小路径和的计算；
```c++
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int m = matrix.size(),n = matrix[0].size();
        vector<vector<int> >dp(m,vector<int>(n+2,0));
        for(int i=0;i<m;i++){
            dp[i][0] = 0x6fffffff;
            dp[i][n+1] =0x6fffffff;
        }
        //每一行左右各补充一个大数，避免边界判断(因为大数加上当前元素一定不会出现在最小下降路径和上)
        for(int j=1;j<=n;j++)
            dp[0][j] = matrix[0][j-1];//dp第一行就是原矩阵的第一行
        for(int i=1;i<m;i++){
            for(int j=1;j<=n;j++){
                int tmp =min(dp[i-1][j-1]+matrix[i][j-1],dp[i-1][j]+matrix[i][j-1]);
                dp[i][j] = min(tmp,dp[i-1][j+1]+matrix[i][j-1]);
                //利用tmp进行三个数的最小值比较
            }
        }
        int ans = 0x7fffffff;
        for(int j=1;j<=n;j++)
            ans = min(ans,dp[m-1][j]);//在最后一行里找最小值作为结果
        return ans;
    }
};
```

[221.最大正方形](https://leetcode.cn/problems/maximal-square/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
关联题：[1277.统计全为1的正方形子矩阵](https://leetcode.cn/problems/count-square-submatrices-with-all-ones/description/) 见下一题；
本体的状态转移方程需要总结，涉及找子矩阵；
令$dp[i][j]$为以(i,j)为右下角，满足条件的正方形的边长的最大值，则：判断原数组$matrix[i][j]$是否为1：
1.如果为0，则$dp[i][j]=0$；
2.如果为1，$dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1$,即判断左侧，上侧，左上侧3个位置的dp值；
同时考虑边界情况,即第一行，第一列如果原数组为1，则对应的dp最大只为1；
遍历所有dp值，返回最大值的平方；
```c++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size(),n = matrix[0].size();
        vector<vector<int> > dp(m,vector<int>(n,0));
        for(int i=0;i<m;i++)
            matrix[i][0]=='1'?dp[i][0]=1:dp[i][0]=0;
        for(int j=0;j<n;j++)
            matrix[0][j]=='1'?dp[0][j]=1:dp[0][j]=0;
        int ans = 0;
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[i][j]=='1'){
                    int tmp = min(dp[i-1][j],dp[i][j-1]);
                    dp[i][j] =min(tmp,dp[i-1][j-1])+1;
                }
            }
        }
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                ans = max(ans,dp[i][j]);
        return ans*ans;
    }
};
```

[1277.统计全为1的正方形子矩阵](https://leetcode.cn/problems/count-square-submatrices-with-all-ones/description/)
思路：
根据上一题221，本题在计算出dp数组后，对dp数组求和，即为结果；
对dp数组求和理解：$dp[i][j]$不仅表示以(i,j)为右下角的正方形的最大边长，也表示以(i,j)为右下角的正方形的数目(即以(i,j)为右下角,边长为1,2,.$dp[i][j]$的正方形各一个),所以对dp数组累加即求得所有以任意(i,j)为右下角的，边长各式的正方形的数目.--因为固定了右下角，所以以(i,j)为右下角的小正方形不会重复(类比双指针固定right，收缩遍历left，结果不会重复)。
本题主要关注题解中推导的上一题的动态规划转移方程；
```c++
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        int m = matrix.size(),n = matrix[0].size();
        vector<vector<int> > dp(m,vector<int>(n,0));
        for(int i=0;i<m;i++)
            dp[i][0] = matrix[i][0];
        for(int j=0;j<n;j++)
            dp[0][j] = matrix[0][j];
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++)
                if(matrix[i][j]==1){
                    int tmp = min(dp[i-1][j],dp[i][j-1]);
                    dp[i][j] = min(tmp,dp[i-1][j-1])+1;
                }
        }
        int ans =0;
        for_each(dp.begin(),dp.end(),[&](const auto& d){
            ans += accumulate(d.begin(), d.end(),0);
        });//对dp每一行进行求和，即对dp整个二维数组求和
        return ans;
    }
};
```

[5.最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：

```c++

```

[516.最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/description/?envType=study-plan-v2&envId=dynamic-programming)
本题需要总结：$dp[i][j]$表示字符串s下标范围$[i:j]$最长回文子序列的长度；
[区间DP](https://www.bilibili.com/list/watchlater?oid=951982456&bvid=BV1Gs4y1E7EU&spm_id_from=333.1007.top_right_bar_window_view_later.content.click)
区间DP会把问题规模缩小到数组中间的区间上，而不仅仅是前缀或后缀；
思路：
对于一个子序列而言，如果是回文序列，且长度大于2，则将其首尾两个字符去除后，仍然是回文子序列；dp数组定义见上，则：
1.$dp[i][i]=1$，因为任何长度为1的子序列都是回文子序列；
2.如果$s[i]==s[j]$,则$dp[i][j]=dp[i+1][j-1]+2$；
3.如果$s[i]!=s[j]$,则$dp[i][j]=max(dp[i+1][j],dp[i][j-1])$,因为$s[i]$和$s[j]$不会是同一个回文子序列的首尾字符；
4.注意for循环中ij的大小变化，原因见注解；
5.最终$dp[0][n-1]$中的值即为s中最长回文子序列的长度；

对于第3点，例子如下：
如字符串：s="abbac",计算$dp[0][4]$时，发现'a'!='c',所以'a'和'c'不是同一个回文子序列的首尾字符，所以$dp[0][4]$只能从$dp[0][3]$和$dp[1][4]$中取最大值，即s$[0:4]$中的最长回文子序列长度为：4;
```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int len = s.size();
        vector<vector<int> > dp(len,vector<int>(len,0));
        for(int i=len-1;i>=0;i--){
            //i从右到左，因为下面涉及dp[i+1],如果从i=0开始，则dp[i+1]还没有计算到，不符合动态规划
            dp[i][i] = 1;
            char c1 = s[i];
            for(int j=i+1;j<len;j++){
            //j相当于长度遍历，从长度为2,3,..,len-1(最长情况下即i=0)
            //且因为dp[i][j-1],从j-1转移过来，所以j要正序枚举
                char c2=s[j];
                if(c1==c2){
                    dp[i][j] = dp[i+1][j-1]+2;
                }
                else{
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][len-1];
    }
};
```

[1039.多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/description/)
思路：

```c++

```

[1143.最长公共子序列-LCS](https://leetcode.cn/problems/longest-common-subsequence/description/)
思路：
子序列本质就是选或不选；考虑$s[i]$和$t[j]$选或不选；
$dp[i][j]=max(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+(s[i]==t[j])$
进而:
--思考为什么$s[i]==s[j]$的时候可以不考虑$dp[i][j-1]$和$dp[i-1][j]$:
```c++
讨论：dp[i-1][j]和dp[i-1][j-1]的情况，dp[i][j-1]类似：

s:abcdc
t:abc

令dp(i-1,j-1)为x,此时x=2,此时：
s:abcd
t:ab

假设:
dp(i-1,j)>x+1(即假设dp(i-1,j)更优):
--两都选‘c’则lcs+1,即x+1;只选一个,即dp(i-1,j)因为假设更优,所以dp(i-1,j)>x+1,即(s:abcd,t:abc)的情况下lcs比dp(s:abcd,t:ab)+1大;
此时dp(i-1,j)为：
s:abcd
t:abc

在上一基础上，去掉匹配的c(使得不等式两边的lcs都减1),如下:
s:abd
t:ab
此时s和t的lcs>x;

因为s:abd和t:ab是：
s:abcd
t:ab
的子序列，所以s:abd 和 t:ab的lcs<=x
矛盾
所以dp(i-1,j)<=x+1,所以更优的是dp(i-1,j-1)+1;
所以在s[i]==t[j]的时候，只用考虑都选的情况:dp[i-1][j-1],令其+1;
```
--思考为什么$s[i]!=s[j]$的时候可以不考虑$dp[i-1][j-1]$
```c++
dp(i,j)由dp(i-1,j)和dp(i,j-1)得来，又从而由dp(i-1,j-1)得来，即:
1.如果dp(i-1,j)不选t[j],则递归到dp(i-1,j-1)
2.如果dp(i,j-1)不选s[i],则递归到dp(i-1,j-1)
即dp(i-1,j-1)的结果已经包含在dp(i-1,j)和dp(i,j-1)中;
即dp(i-1,j)>=dp(i-1,j-1);dp(i,j-1)>=dp(i-1,j-1)
```
$dp[i+1][j+1]=dp[i][j]+1,s[i]==s[j]$；
$dp[i+1][j+1]=max(dp[i][j+1],dp[i+1][j]),s[i]!=s[j]$；
其中，$dp[i][j]$表示$s[0:i]$和$t[0:j]$的最长公共子序列的长度;--前i个和前j个的LCS
边界情况:$dp[0][j]$和$dp[i][0]$为0；
最终结果存储在：$dp[n][m]$中，n为s长度，m为t长度；
```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size();
        int m = text2.size();
        vector<vector<int> >dp(n+1,vector<int>(m+1,0));
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(text1[i]==text2[j])
                    dp[i+1][j+1] = dp[i][j]+1;
                else
                    dp[i+1][j+1] = max(dp[i][j+1],dp[i+1][j]);
            }
        }
        return dp[n][m];
    }
};
```

[72.编辑距离](https://leetcode.cn/problems/edit-distance/description/?envType=study-plan-v2&envId=dynamic-programming)
总结：编辑距离算法是一种用于衡量两个字符串之间相似度的算法，返回从一个字符串转换到另一个字符串所需的最小操作数；被用作机器翻译和语音识别评价标准；NLP中编辑距离算法可以用于拼写纠错和文本相似度计算；

思路：
可以参考题：1143，
if：$s[i]==t[j]$：
$dp[i][j]=dp[i-1][j-1]$；
else if：$s[i]!=t[j]$：
$dp[i][j]=min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1$,对应:插入,删除,替换
```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        vector<vector<int> >dp(n+1,vector<int>(m+1,0));
        
        for(int j=0;j<=m;j++)
            dp[0][j] = j;
            //即若一个字符串为空，则要把另一个字符串都去掉;
            
        for(int i=0;i<n;i++){
            dp[i+1][0] = i+1;
            //同上，若一个字符串为空，则把另一个字符串都去掉;
            //下标为0~i的字符串长度为i+1
            for(int j=0;j<m;j++){
                if(word1[i]==word2[j])
                    dp[i+1][j+1] = dp[i][j];
                else{
                    int tmp = min(dp[i][j+1],dp[i+1][j]);
                    dp[i+1][j+1] =min(tmp,dp[i][j])+1;
                }
            }
        }
        return dp[n][m];
    }
};
```

[300.最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)
思路1:
动态规划，定义dp(i)为以num(i)结尾的最长递增子序列的长度，则状态转移方程可以为：当前元素之前比当前元素num(i)小且dp最大的值+1；
```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        int dp[len];
        fill(dp,dp+len,1);
        
        for(int i=1;i<len;i++){
            for(int j=i-1;j>=0;j--){
                if(nums[j]<nums[i]){
                    dp[i] = max(dp[i],dp[j]+1);
                }
            }
        }
        return *max_element(dp, dp+len);
    }
};
```

[583.两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/description/)
思路：

```c++

```

[139.单词拆分](https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2&envId=dynamic-programming)
思路：
本题需要总结dp数组的定义方式；
定义$dp[i]$为字符串s前i个字符构成的子串$s[0:i-1]$是否能被空格拆分成<u>若干个</u>字典中出现的单词；枚举$s[0:i-1]$的分割点j,判断$s[0:j-1]$和$s[j:i-1]$是否能由若干个wordDict中的单词拼出来；因为$s[0:j-1]$即$dp[j]$已经判断过了(因为j比i小)，所以只需要判断后一部分是否出现在字典中即可；
--注意,j=0时s1是空串,即判断整个不做切割的子串$s[0:i-1]$是否在wordDict中

```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int len = s.size();
        vector<bool> dp(len+1,0);//dp[i]代表前i个字符，所以n+1个
        set<string>st;//便于查找word是否在wordDict中
        for_each(wordDict.begin(),wordDict.end(),[&](const string& s){
            st.insert(s);
        });
        //dp[i]表示字符串前i个字符组成的字符串s[0..i-1]能否被空格拆分成若干字典中出现的单词
        dp[0] = true;//前0个字符，空字符一定能拼出
        for(int i=1;i<=len;i++){//前i个字符，即s[0:i-1]
            for(int j=0;j<i;j++){
            //前j个字符，即s[0:j-1],j==i和j==0情况重复，所以j最大取i-1
                //判断如果前j个可以拼出，且s[j:i)在字典中,则dp[i]=true
                if(dp[j] && st.find(s.substr(j,i-j))!=st.end()){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[len];
    }
};
```
利用trie树(字典树)实现判断str(j,i-j)是否在wordDict中：
```c++
struct TrieNode{
    bool isword;
    TrieNode* child[26];
    TrieNode(){
        isword = false;
        for(int i=0;i<26;i++)
            child[i] = NULL;
    }
};
class Trie{
    private:
        TrieNode* root;
    public:
        Trie(){
            root = new TrieNode();
        }
    void insert(const string& s){
        int len = s.size();
        TrieNode* cur = root;
        for(int i=0;i<len;i++){
            int c = s[i]-'a';
            if(cur->child[c]==NULL){
                cur->child[c] =new TrieNode();
            }
            cur = cur->child[c];
        }
        cur->isword = true;
    }
    bool search(const string& s){
        int len = s.size();
        TrieNode* cur = root;
        for(int i=0;i<len;i++){
            int c = s[i]-'a';
            if(cur->child[c]==NULL)
                return false;
            cur = cur->child[c];
        }
        return cur->isword;//判断此时这个字符是不是某个字符串的结尾
    }
};
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int len = s.size();
        vector<bool> dp(len+1,0);
        Trie* t = new Trie();
        for_each(wordDict.begin(), wordDict.end(), [&](const string& word){
            t->insert(word);
        });
        //dp[i]表示字符串前i个字符组成的字符串s[0..i-1]能否被空格拆分成若干字典中出现的单词
        dp[0] = true;//前0个字符，空字符一定能拼出
        for(int i=1;i<=len;i++){//前i个字符，即s[0:i-1]
            for(int j=0;j<=i;j++){//前j个字符，即s[0:j-1]
                //判断如果前j个可以拼出，且s[j:i)在字典中,则dp[i]=true
                if(dp[j] && t->search(s.substr(j,i-j))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[len];
    }
};
```

## 图论
[图论 · 从入门到精通](https://leetcode.cn/studyplan/graph-theory/)
相关知识：
1.设图G为无向图，若G中任意两点之间都有道路，则称G为连通图，反之非连通图；
--连通图图中任意两点之间都有路径可达；--连通图是无向图中的概念

2.设图G为无向图，G的极大连通子图称为连通分量;
--连通图的连通分量只有1个,即其自身；

3.设图G为有向图,若G中任意两点间都存在相互可达的路径,则称G为强连通图；
--强连通图中任意两点要求相互可达；--强连通图是有向图中的概念

4.设图G为有向图，G的极大强连通子图称为G的强连通分量；
--强连通图的强连通分量只有1个，即其自身；

5.完全图：如果一个图中任意两点间有边(有向图不一定相互可达)，则为完全图

[547.省份数量](https://leetcode.cn/problems/number-of-provinces/description/?envType=study-plan-v2&envId=graph-theory)
思路：
法1:从每个未标记节点开始进行DFS遍历，同时标记遍历过的节点，每进行一次DFS则ans++；即求解连通分量的个数(极大连通子图的个数)；
```c++
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        bool ismarked[n];
        int ans = 0;
        memset(ismarked,0,sizeof(ismarked));
        auto dfs = [&](auto&& dfs,int node)->void{
            for(int i=0;i<n;i++){
                if(ismarked[i]==false && isConnected[i][node]){
                    ismarked[i] = true;
                    dfs(dfs,i);
                }
            }
            return;
        };
        for(int i=0;i<n;i++){
            if(ismarked[i]==false){
                dfs(dfs,i);
                ans++;
            }      
        }
        return ans;
    }
};
```

法2:利用并查集，将属于一个集合的元素连接到同一个根结点，最后用map遍历每个结点的根结点，返回map的size(key的数量)即为省份的个数；
```c++
class unionFind{
private:
    vector<int> parent;
    //parent数组,即parent[i]=j即i的父结点是j
    vector<int>rank;
    //rank数组,集合的秩，即树高
public:
    unionFind(int n){
        parent.resize(n);
        rank.resize(n,1);
        //初始时每个结点的树高度为1
        for(int i=0;i<n;i++)
            parent[i] = i;
        //初始化n个元素，下标为0~n-1，同时让每个元素的父结点指向自己
    }
    int find(int x){
        if(parent[x]!=x){
            parent[x] = find(parent[x]);
        }//路径压缩
        return parent[x];
    }//返回的是x的根结点编号
    void unionSet(int x,int y){
        int rootX = find(x);
        int rootY = find(y);
        if(rootX!=rootY){
            if(rank[rootX]>rank[rootY]){
                parent[rootY] = rootX;
            }else if(rank[rootX]<rank[rootY]){
                parent[rootX] = rootY;
            }
            else{
                //如果树高相同，则随机让rootX指向rootY，同时使rootY的树高+1
                parent[rootX] = rootY;
                rank[rootY]++;
            }
        }
    }
    bool isConnected(int x,int y){
        return find(x)==find(y);
    }
};
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        unionFind uf(n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(isConnected[i][j])
                    uf.unionSet(i, j);
            }
        }
        map<int, int>cnt;
        for(int i=0;i<n;i++){
            cnt[uf.find(i)]++;
        }
        return cnt.size();
    }
};
```


## 博弈
[game-theory](https://leetcode.cn/problem-list/game-theory/)

[ 运动饮料和矿泉水](https://www.lanqiao.cn/problems/19847/learning/?contest_id=206)
问题描述：
```
为了让运动员们保持最佳状态，学院准备了大量的运动饮料🥤和矿泉水💧。
操场上，一排长长的桌子上一共摆放了N瓶水，其中从左往右奇数位置上的水都是能够补充能量的运动饮料，而偶数位置上的水则都是清爽解渴的矿泉水。现在，有两个班级的代表来取水，他们将轮流从桌子的一端（最左侧或最右侧）取走一瓶水，直到桌子上的水被全部取完。这两个班级的代表都鬼精鬼精的，都想为自己的班级取走尽可能多的运动饮料。请问，如果这两个代码都采取最佳策略，那么最终优先取水的代表最多可以拿到多少瓶运动饮料呢？
N (1≤N≤10^5)，输出一个整数，表示优先取水的代表最多可以获得的运动饮料瓶数;
```
思路：

```c++

```

## 回溯
增量构造答案的过程，这个过程用递归实现；不用想太细(怎么往下递),写对边界条件和非边界条件的逻辑即可（交给数学归纳法）。
回溯三问：当前操作是什么；子问题是什么；下一个子问题是什么；

[17.电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/)
本题总结：如何使用匿名函数进行递归；
当前操作：枚举$path[i]$要填入的字母
子问题：构造字符串$>=i$的部分
下一个子问题：构造字符串$>=i+1$的部分
```c++
匿名函数写法：
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        int n  = (int)digits.size();
        if(n==0)return vector<string>();
        vector<string>ans;
        string path(n,0);//n长度的0
        string l[10] = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        auto dfs = [&](auto&& dfs,int step){
            if(step==n){
                ans.push_back(path);
                return;
            }
            for(char c:l[digits[step]-'0']){
                path[step] = c;
                dfs(dfs,step+1);
            }
            
        };
        dfs(dfs,0);
        return ans;
    }
};

非匿名函数写法：
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        int n = digits.size();
        if(n==0)
            return {};
        vector<string>ans;
        string l[10] = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        string tem="";
        dfs(0,n,l,ans,tem,digits);
        return ans;
    }
    void dfs(int step,int n,const string l[10],vector<string>&ans,string cur,string& digits){
        if(step==n){
            ans.push_back(cur);
            return;
        }
        for(char c:l[digits[step]-'0']){
            dfs(step+1,n,l,ans,cur+c,digits);
        }
    }
};
```

[78.子集](https://leetcode.cn/problems/subsets/description/)
本题总结：
如何生成一个给定数组的子集；
选/不选问题，
如果不选：则直接进入下个递归；
如果选：则步骤为：选中，递归，恢复现场

当前操作：枚举第i个数选/不选；
子问题：从下标$>=i$的数中构造子集；
下一个子问题：从下标$>=i+1$的数中构造子集；
```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int> >ans;
        vector<int> tmp;
        auto dfs = [&](auto&& dfs,int step,int l){
            if(step==l){
                ans.push_back(tmp);
                return;
            }
            dfs(dfs,step+1,l);
            tmp.push_back(nums[step]);
            dfs(dfs,step+1,l);
            tmp.pop_back();
            //恢复现场，可以绘制搜索树进行查看，只有恢复现场才能回到上一个父结点进行下一个选择操作的搜索；
        };
        dfs(dfs,0,n);
        return ans;
    }
};
```

[131.分割回文串](https://leetcode.cn/problems/palindrome-partitioning/description/)
思路：
从分割字符串s转变为是否选择切割字符串的逗号，如"aab",则看作"a,a,b,",对每个逗号都判断选或不选，然后判断选取逗号后切出的子串是否是回文串即可；
```c++
class Solution {
public:
    bool judge(string& s,int begin,int end){
        int i = begin,j =end;
        while(i<j){
            if(s[i]!=s[j])
                return false;
            i++;j--;
        }
        return true;
    }
    vector<vector<string>> partition(string s) {
        int n = s.size();
        vector<string>path;
        vector<vector<string> > ans;
        
        auto dfs = [&](auto&& dfs,int i,int start){
            //start表示当前回文串开始的位置
            if(i==n){
                ans.push_back(path);
                return;
            }
            if(i<n-1){
                dfs(dfs,i+1,start);
            }//不选i和i+1之间的","，i==n-1时，即最后一个分割必须要选
            if(judge(s, start, i)){
                path.push_back(s.substr(start,i-start+1));
                dfs(dfs,i+1,i+1);
                path.pop_back();
            }
        };
        dfs(dfs,0,0);
        return ans;
    }
};
```



## 每日一题
[2187.完成旅途的最少时间](https://leetcode.cn/problems/minimum-time-to-complete-trips/description/)
本题可以总结：使用匿名函数进行代码简化；二分查找写法(左右闭区间)；
思路：
首先判断花费t时间能否完成totalTrips趟旅途。如果对某个t判定成立，则对区间$[t,+\infty]$内所有整数也成立。这也就说明这个判定问题对于花费时间t具有二值性。所以用二分查找确定使得该判定问题成立的最小t；
```c++
class Solution {
public:
    long long minimumTime(vector<int>& time, int totalTrips) {
        auto check = [&](long long t)->bool{
            long long cnt = 0;
            for(const auto& period:time)
                cnt += t/period;
            return cnt>=totalTrips;
        };//传入一个时间t，判断在时间t下能完成的总旅途趟数cnt是多少，返回这个总旅途趟数cnt能不能达到要求的totalTrips;
        long long l = 1;//二分查找的下界，时间最小是1
        long long r = (long long)totalTrips*(*max_element(time.begin(), time.end()));
        //二分查找上界，时间最大不超过最大time*总共旅途数（用最小time*总共旅途数也可）
        while(l<=r){
            long long mid = l+(r-l)/2;
            if(check(mid)){//如果当前时间仍能完成总旅途数，就缩短时间
                r = mid-1;
            }else l = mid+1;//否则增大时间
        }
        return l;
    }
};
```

[134.加油站](https://leetcode.cn/problems/gas-station/description/)
思路：
观察数据量，因为$n<=10^5$,所以如果简单模拟过程一定会超时；
贪心解法：
```c++
gas:  2 5 2 3 5
cost: 1 2 8 2 4
minus:1 3 -6 1 1
其中，minus[i] = gas[i]-cost[i];
发现：如果从minus[0]出发，则达到minus[2]一定会负油，所以从index=0,1,2出发都不可以；如果发现curSum为负(即1+3-6=-2，index=2)，则从下一个位置出发；

一个可能的疑问：从[i:j]到达j位置时curSum<0,有没有可能从[i...,k...,j]，即不从i出发，而从k位置出发能使得到j时curSum>=0?
假设：如果可能，则划分区间:s1=[i:k-1];s2=[k:j],其中，s2的curSum>=0，而s[i:j]的curSum<0,所以s1[i:k-1]的curSum<0,即此时k的位置也就是curSum为负的下一个位置，和‘发现’中指出的情况一致；

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        vector<int> minus(n,0);
        for(int i=0;i<n;i++)
            minus[i] = gas[i]-cost[i];
        
        int curSum=0;//看当前油量总和是否为负，如果为负，则从当前的下一个位置开始，同时curSum=0
        int totalSum = 0;//对minus进行加和，如果为负，则说明从任何位置开始都不可能跑完一圈
        int idx = 0;//记录开始的位置，即最终返回的答案
        for(int i=0;i<n;i++){
            curSum+=minus[i];
            totalSum+=minus[i];
            if(curSum<0){
                curSum = 0;
                idx = (i+1)%n;
            }
        }
        if(totalSum<0)
            return -1;
        else
            return idx;
    }
};
```

[287.寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/description/)
思路：
1.暴力求解:使用集合set，查找当前元素x，如果找到，则直接返回，否则加入st
2.要求不修改数组且只用常量O(1)的额外空间：
a)弗洛伊德判圈法(快慢指针法)；b)二分查找法；c)二进制法；

a)弗洛伊德判圈法：
1.可以在有限状态机、迭代函数、链表上判断是否有环、环的起点和长度；
2.过程：
对nums数组建图，即$i->nums[i]$,因为target重复，所以位置i=target至少有两条边指向它，因此图里存在环；

接下来使用快慢指针和弗洛伊德判圈法即可解决；--见本题下方笔记；
```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0,fast = 0;
        do{
            slow = nums[slow];
            fast = nums[fast];
            fast = nums[fast];
        }while(slow!=fast);
        int head = 0;
        while(head!=slow){
            head = nums[head];
            slow = nums[slow];
        }
        return slow;
    }
};
```

b)二分查找法：
定义$cnt[i]$表示nums中小于等于$i$的数有多少个，假设重复的是target，则：
$[1,target-1]$中$cnt[i]<=i$
$[target,n]$中$cnt[i]>i$
即，cnt数组随下标i增大，具有单调性；---使用二分查找
例如：$[1,3,4,4,4]$,其中$cnt[3]=2<3$；$[1,2,3,4,4]$,其中区间$[1,target-1]$中$cnt[i]==i$,区间$[target,n]$中$cnt[i]>i$；
--重复数字是否超过2个的区别：重复个数超过2，则相当于用target取代没出现的数字；否则重复个数等于2，即鸽巢原理：如果有n个抽屉和(n+1)个物体，那么至少有一个抽屉里会有两个物体；
```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int len = nums.size();
        int l =1,r = len-1;
        int ans = -1;
        while(l<=r){
            int mid = l+(r-l)/2;
            int cnt = 0;
            for(int i=0;i<len;i++){
                cnt += (nums[i]<=mid);
            }
            if(cnt<=mid)
                l = mid+1;
            else{
                r = mid-1;
                ans = mid;
            }
        }
        return ans;
    }
};
```

c)二进制：
将数组中的每个数按照二进制各位展开(如：3->011),确定重复数每一位是1还是0，就可以按位还原重复数；
考虑第i位(从低到高从0~log2(n))：
记nums数组中第i位为1的个数为：x
记数字$[1,n]$中第i位为1的个数为：y
则：重复的数第i位为1当且仅当x>y;
即结果为:
$([0:i-1]=0)$+$([i]=1)$+$([i+1,log2(n)]=0)$
```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int high = (int)log2(n);
        int ans = 0;
        for(int bit=0;bit<=high;bit++){
            int x=0,y=0;//统计每一个nums[i]中第bit位的1的个数
            for(int i=0;i<n;i++){
                //[0,n-1]共n个为数组大小，[1,n-1]共n-1个为每个nums[i]的取值范围
                if(nums[i]&(1<<bit))//注意，用&操作判断某位是否为1，不要写==1，结果非0对应位就是1
                    x++;
                if(i>=1 && ((i&(1<<bit))))
                    y++;
            }
            if(x>y)
                ans |= (1<<bit);
                //常规，通过｜操作和移位操作拼接二进制位
        }
        return ans;
    }
};
```

快慢指针相关：
[环形列表](https://www.bilibili.com/video/BV1KG4y1G7cu/?spm_id_from=333.999.0.0&vd_source=3ac79ed435a4827c66109984966d124a)

[876.链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/description/)
思路：
快慢指针，每次快指针走2步，慢指针走1步；当：
1.链表结点个数为偶数时，fast到空结点时slow到链表的中点；
2.链表结点个数为奇数时，fast到最后一个结点时slow到链表的中点；
综上所述，当`fast!=NULL&&fast->next!=NULL`才进入循环，否则返回slow；
```c++
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* fast=head,*slow=head;//设置快慢指针
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
};
```

[141.环形链表](https://leetcode.cn/problems/linked-list-cycle/description/)
思路：
同题876，使用快慢指针，如果链表存在环，则当快慢指针进入环后必定会在环中某个位置相遇，否则代码逻辑将同876一致，当退出循环时slow指向链表中点
```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast=head,*slow=head;//设置快慢指针
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
            if(fast==slow)
                return true;
        }
        return false;
    }
};
```

[142.环形链表II](https://leetcode.cn/problems/linked-list-cycle-ii/description/)
思路：
本题大概意思同141一致，不过如果链表中有环，需要返回这个环的入口；
```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast=head,*slow=head;//设置快慢指针
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
            if(fast==slow){
            //当快慢指针在环中相遇时，用一个新指针从头开始移动，同时slow也同步移动，当slow和head相遇时，即为环的入口
                while(head!=slow){
                    head = head->next;
                    slow = slow->next;
                }
                return slow;
            }
        }
        return NULL;
    }
};
```

[206.反转链表](https://leetcode.cn/problems/reverse-linked-list/description/)
思路：
三个指针:pre,cur,nxt分别初始化为：null,head,head->next;
使得：
nxt = cur->next;//暂存cur->next
cur->next = pre;
pre = cur;
cur = nxt;
相当于每次循环仅添加一条从cur到pre的边，然后对3个指针依次向后移动一个结点；最后当cur为空时，pre指向原链表的最后一个结点，返回pre即为head';
```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = NULL;
        ListNode* cur = head;
       
        while(cur){
            ListNode* nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
};
```
[92.反转链表II](https://leetcode.cn/problems/reverse-linked-list-ii/description/)
本题要总结：反转链表中的一段，总结dummy结点；
如：L:$n_0,n_1,n_2,n_3,n_4,n_5$，反转2~4段，即：L':$n_0,n_1,n_4,n_3,n_2,n_5$
`   d  h          p cur`
根据题206，当反转结束时，pre = $n_4$,cur=$n_5$;p0=$n_1$
当left=0时，没有p0，所以设置p0=dummy,使得dummy->next=head;
反转中间这一段后，使：
```c++
# 伪代码如下：
dummy  = new Node()
dummy->next = head;//即原头结点前定义一个哨兵结点
p0 = dummy;
for(int i=0;i<left-1;i++)
	p0 = p0->next;//使得p0来到反转链表段开头的前一个结点

pre = NULL;
cur = p0->next;
for(int i=0;i<left-right+1;i++){
	nxt = cur->next;
	cur->next = pre;
	pre = cur;
	cur = nxt;	
}//反转中间的right-left+1个元素

p0->next->next = cur
p0->next = pre
return dummy->next;
```
具体实现如下：
```c++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head,int left,int right) {
	    //[left,right]左右都闭
        ListNode* dummy = new ListNode(-1,head);
        //dummy在head前，dummy->next = head;
        ListNode* p0 = dummy;
        
        for(int i=0;i<left-1;i++){
            p0 = p0->next;
        }//找到翻转段的前一个位置
        ListNode* cur = p0->next;
        ListNode* pre = NULL;
        for(int i=0;i<right-left+1;i++){
            ListNode* nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }//这一部分即题206
        p0->next->next = cur;
        p0->next = pre;
        return dummy->next;
    }
};
```

[25.k个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/)
思路：

```c++

```


[143.重排列表](https://leetcode.cn/problems/reorder-list/description/)
思路：
要求L:$n_0,n_1,n_2,...n_n$变为L':$n_0,n_n,n_1,n_{n-1},n_2,n_{n-2}...$
解法：找到链表L的中点，翻转从中点开始到结尾的这段链表，然后用两个指针分别从第一段的head和第二段的head=mid开始，执行交叉连接，因为用到之后需要next移动到下一个结点，所以用临时变量提前保存各自的next；

需要注意的是，翻转了从mid到结尾的链表后，如1234->1243,2的next指针指向的是3,3的next指针指向NULL；
--即前一段的最后一个结点一定连在后一段的最后一个元素上；
```c++
class Solution {
public:
    void reorderList(ListNode* head) {
        ListNode* mid = findMid(head);
        //找到链表中点
        ListNode* seg2 = reverseList(mid);
        //翻转从中间到最后的一段链表
        ListNode* seg1 = head;
        
        while(seg2->next){
            ListNode* nxt1 = seg1->next;
            ListNode* nxt2 = seg2->next;
            seg1->next = seg2;
            seg2->next = nxt1;
            
            seg1 = nxt1;
            seg2 = nxt2;
        }
    }
    //用于找链表的中点
    ListNode* findMid(ListNode* head){
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast && fast->next){
            fast = fast->next->next;
            slow = slow->next;
        }
        return slow;
    }
    //用于翻转一段链表
    ListNode* reverseList(ListNode* head){
        ListNode* pre = NULL;
        ListNode* cur= head;
        while(cur){
            ListNode* nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
};
```


[871.最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/description/)
总结：一个比较好的思想是，不要每次跟着题目去模拟，可以先把数据存着，当后面条件不满足时再去存着的数据里进行操作；像本题不要去模拟加油站加油，而是把加油站的油带着走，直到后面curfuel不满足抵达新加油站/终点时，再去看最大堆中的数据；

思路：
首先需要注意，stations数组已经按照$station[i][0]$进行非递减排序，即按照到起点的距离排序，所以遍历到一个加油站i时，所有小于i的加油站都应被遍历过；

思考方式：当汽车到达一个加油站i时，视作顺走了装有所有油的油箱fuel_i；在后续移动中，如果curfuel<0,则不断从前面顺走的油箱中按照每次选取最大油箱的规则加油(看作拜访了一个加油站，使得ans++)；

用贪心每次加油都选择最大的油箱，以减少加油的次数；--可以使用优先队列即最大堆；

将终点视作油箱为0的虚拟加油站，加到station末尾，这样$station[n-1]$到终点target这一段距离就不用单独处理了；
```c++
//如果不加入虚拟加油站，则还要处理最后一个加油站到终点这一段路：
curfuel-=(target-pre_position);//即最后一段的pos=target
while(!q.empty() &&curfuel<0){
	curfuel += q.top();
	q.pop();
	ans++;
}
if(curfuel>=0)
	return ans;
else return -1;
```

可以一开始就判断startfuel>=target，则直接返回0；
```c++
class Solution {
public:
    int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations) {
        if(startFuel>=target)
            return 0;
        
        stations.push_back({target,0});
        //将终点视作一个加油为0的虚拟加油站
        
        priority_queue<int>q;
        int ans=0;//最小加油次数
        int curfuel=startFuel;//现在邮箱里的油量
        int pre_position=0;//上一段走过的距离,即station[i-1][0]
        for(const auto& station:stations){
            int pos = station[0];//距离
            curfuel -= (pos-pre_position);//计算当前油量，即：当前station的距离-上一个station的距离
            while(!q.empty() && curfuel< 0){
                curfuel+=q.top();//油量为负，就用之前加油站顺的油箱，每次加最多的油
                q.pop();
                ans++;//到过的加油站数+1
            }//如果没油了，则从能走到的加油站里加油
            if(curfuel<0){
                return -1;
            }//如果油还是不够(之前到过的加油站顺的油箱都用完了还没加够)返回-1，即到不了
            q.push(station[1]);//如果能到这个加油站，顺走油箱，留着后面加油，即本题是把所有能到的加油站的油都带走，要加油的时候就ans++
            pre_position = pos;
        }
        return ans;
    }
};
```

[1436.旅行终点站](https://leetcode.cn/problems/destination-city/description/)
思路：
1.暴力：利用邻接表存储，如果表中某一个点的出边为0，则该点就是终点；可以选择map或multimap作为实现的数据结构,即：
$map<string,vector<string> >mp$或$multimap<string,string>mlp$
2.尝试使用2个哈希set实现一次遍历：对于路径$<a,b>$ ,a一定不是终点，b是终点的前提是b不在a中出现，所以使用2个jiji
```c++
class Solution {
public:
    string destCity(vector<vector<string>>& paths) {
        unordered_set<string>st_a;
        unordered_set<string>st_b;
        int len = paths.size();
        for(int i=0;i<len;i++){
            string a_i = paths[i][0];
            string b_i = paths[i][1];
            st_b.erase(a_i);
            if(!st_a.contains(b_i)){
                st_b.insert(b_i);
            }
            st_a.insert(a_i);
        }
        return *st_b.begin();
    }
};
```

[3171.找到按位或最接近k的子数组](https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/)
思路：

```

```

[3162.优质数对的总数I](https://leetcode.cn/problems/find-the-number-of-good-pairs-i/)
相关题目：[3164.优质数对的总数II](https://leetcode.cn/problems/find-the-number-of-good-pairs-ii/description/)
思路：
1.模拟，按照题目要求进行模拟；
```c++
class Solution {
public:
    int numberOfPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        int n = nums1.size(),m = nums2.size();
        sort(nums1.begin(),nums1.end());
        sort(nums2.begin(),nums2.end());
        for(int i=0;i<m;i++)
            nums2[i]*=k;
        int ans = 0;
        for(int i=0;i<m;i++){
            auto it = lower_bound(nums1.begin(), nums1.end(),nums2[i]);
            while(it!=nums1.end()){
                if((*it)%nums2[i]==0)
                    ans++;
                it++;
            }
        }
        return ans;
    }
};
```

2.枚举因子：
首先，区分一个概念：a能被b整除等价于：$a\%b=0$,即a有一个因子是b；
所以，如果$a\%(k*b)==0$,则$(a/k)\%b==0$,所以只需要枚举a/k的因子即可；
如果b出现在a/k的因子中，则ans++；
```c++
class Solution {
public:
    long long numberOfPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        unordered_map<int, int> ump;
        //设x是nums1中的元素，y是nums2中的元素
        for(int x:nums1){
            if(x%k)
                //如果nums1中的一个元素x都不能被k整除，则不必统计x的因子，直接跳过
                continue;
            //枚举每一个nums1中元素x/k的因子d
            //如果:x%(k*y)==0,则:(x/k)也能整除y,所以枚举x/k的因子，看看里面有没有y即可
            x /= k;
            for(int d=1;d*d<=x;d++){
                if(x%d)
                    continue;//x不是d的倍数--d不是x的因子
                ump[d]++;
                if(d*d<x)//防止出现d^d=x，重复统计
                    ump[x/d]++;//因子成对出现
            }
        }
        long long ans = 0;
        for(int y:nums2){
            ans += ump.contains(y)?ump[y]:0;
        }
        return ans;
    }
};
```

3.枚举倍数：
补充：调和级数，形如：$(\frac{1}{1}+\frac{1}{2}+\frac{1}{3}+...\frac{1}{m})$的时间复杂度为:$ln(m)$；
证明方法：计算$\int_{1}^{m}\frac{1}{x}dx$
假设：nums2中的元素为y，nums1中的元素为x;
枚举y的若干倍数：y,2y,3y....;然后统计nums1中有多少个$(x/k)==y,2y,...$
思想类似线性筛素数，因为$(x/k)\%y==0$,所以存在一个d使得:$(x/k)=d*y$,遍历y的若干倍数就是在遍历d从d=1,2,3...,并且统计所有(x/k)=$d*y$的个数；
举例如下：
```
nums1=[1,2,4,12],nums2 = [2,4],k=3
设nums1中的元素为x，nums2中的元素为y；
1.x%(k*y)==0转换为:先判断x%k==0?x/=k:pass;--即x要先能整除k；
此时，nums1 = [1/3,2/3,4/3,12/3] = [0,0,1,4],其中x'=x/k;
2.再统计x'==y的若干倍的结果，即：因为此时nums1中是x'=x/k,x'要能整除y,即x'要是y的若干倍；
3.为防止有重复元素重复计算，所以cnt1统计nums1中的重复元素个数，cnt2统计nums2中的重复元素，且遍历cnt2，判断y,2y,3y...u,其中u为cnt1中的最大key(即nums1中的最大x/k);
4.最后累加答案即可；
--注意：统计重复元素是因为nums1和nums2均可以出现重复元素：如num1[3,6],num2[3,3,3]，,k=1,则(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),均符合要求；统计cnt2[3]=3;则最后只需要：3*2=6即可；
```

```c++
class Solution {
public:
    long long numberOfPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        unordered_map<int, int>cnt1;//统计nums1中重复出现元素的个数
        int u = -1;
        for(auto &x:nums1){
            if(x%k==0){
                cnt1[x/k]++;
                u = max(u,x/k);
            }
        }
        if(cnt1.empty())
            return 0;
            
        unordered_map<int, int>cnt2; //统计nums2中重复出现元素的个数
        for(auto& y:nums2){
            cnt2[y]++;
        }
        long long ans=0;
        for(auto& [y,cnt]:cnt2){
            int s = 0;
            for(int i=y;i<=u;i+=y)//遍历y,2y,3y...
            {
                s+=cnt1.contains(i)?cnt1[i]:0;
                //累加每一个y的倍数在nums1中出现次数
            }
            ans+=(long long)s*cnt;
        }
        return ans;
    }
};
```

[3158.求出出现两次数字的XOR值](https://leetcode.cn/problems/find-the-xor-of-numbers-which-appear-twice/description/)
思路：
暴力模拟即可；
总结：
使用二进制数表示集合visit(见下文)；
```c++
class Solution {
public:
    int duplicateNumbersXOR(vector<int>& nums) {
        unordered_map<int, int>ump;
        for_each(nums.begin(),nums.end(),[&](const int& val){
            ump[val]++;
        });
        int ans=0;
        for_each(ump.begin(),ump.end(),[&](pair<int,int> p){
            if(p.second>=2){
                ans ^= p.first;
            }
        });
        return ans;
    }
};
```
一次遍历+O(1)空间解法：
总结：
使用二进制数表示集合visit；
```
集合可以用二进制表示，二进制从低到高第i位为1表示i在集合中，为0表示i不在集合中
集合{0,2,3}可以用二进制数1101(2)表示;
包含非负整数的集合s，可以压缩成：
```
$f(s)=\sum_{i \in s}{2^i}$,即集合{0,2,3} = $2^0+2^2+2^3=13$，即1101(2);

使用集合visit维护遇到的数字，设$x = nums[i]$,如果x没在visit中，则是第一次遇见，将其加入visit；否则是第二次遇见，将其异或加入ans；
```c++
class Solution {
public:
    int duplicateNumbersXOR(vector<int>& nums) {
        long long st=0;
        int ans=0;
        for(int x:nums){
           if((st>>x) & 1)
               ans^=x;
               //如1101(2)判断x=3是否在集合里：1101->0001&1==1
            else
                st|=((long long) 1)<<x;
                //注意int类型移动最多31位，即(1<<31)
                //long long类型最多移动63位：(((long long)1)<<63)
        }
        return ans;
    }
};
```

[1884.鸡蛋掉落-两枚鸡蛋](https://leetcode.cn/problems/egg-drop-with-2-eggs-and-n-floors/)
总结：记忆化搜索(递归搜索+保存递归返回值)；
思路：
假设第一枚鸡蛋在第j楼扔下，若：
1.碎了，则从1,2,...j-1扔下第二枚鸡蛋，总共两枚鸡蛋最多需要扔j-1+1=j次；
2.没碎，则第一枚鸡蛋从n-j楼层再去看，总共次数为：dfs(n-j)+1；其中1是这次；即等价于从有n-j层的楼扔鸡蛋的子问题；

为了保证无论在哪种情况下都能确定f的值(任何从高于 f 的楼层落下的鸡蛋都会碎 ，从 f 楼层或比它低的楼层落下的鸡蛋都不会碎),上述两种情况取最大值；
枚举j=1～n，分别查看要扔的次数，取其最小值作为结果；

dfs(i)表示从一栋i层的楼扔鸡蛋，能确定f的最小操作次数；
```c++
int memo[1001];
class Solution {
public:
    int twoEggDrop(int n) {
        if(n==0)
            return 0;
        int& res = memo[n];
        if(res)//本题的特殊初值是0，如果遇到0，则意味着是第一次见，否则是计算过的值，直接返回
            return res;
        res = 0x7fffffff;
        for(int j=1;j<=n;j++)
            res = min(res,max(j,twoEggDrop(n-j)+1));
            //备份计算过的值，后续再次遇到可以直接返回（前面传入引用的原因就是可以直接修改对应的memo数组）
        return res;
    }
};
```

[3191.使二进制数组全部等于1的最少操作次数 I](https://leetcode.cn/problems/minimum-operations-to-make-binary-array-elements-equal-to-one-i/description/)
思路：
当$i==0$时，如果：
$nums[i]==0$,将$nums[i]$和$nums[i+1]$和$nums[i+2]$进行翻转；
$nums[i]==1$,不操作，问题变为处理剩余n-1个数的子问题；
一直处理到$i==n-3$时，如果$i==n-2$和$i==n-1$都为1，则返回ans，否则返回-1；即最后2个数必须是1，否则无法处理；

注意：
1.对于同一个 i，操作两次等于没有操作，所以同一个i至多操作一次。注：操作 i 指的是反转 i,i+1,i+2 这三个位置。
2.从左到右操作的过程中，遇到 1 一定不能操作，遇到 0 一定要操作，所以从左到右的操作方式有且仅有一种。
3.对同一个 i 至多操作一次，就可以做到最少的操作次数。

例子：如011，遇到0一定要操作，所以变为100；后续操作类似；
```c++
class Solution {
public:
    int minOperations(vector<int>& nums) {
        int n = nums.size();
        
        int ans = 0;
        for(int i=0;i<n-2;i++){
            if(nums[i]==0){
                nums[i+1]^=1;
                nums[i+2]^=1;
                ans++;
                //翻转
            }//只遍历一次，所以nums[i]可翻可不翻
        }
        return nums[n-1]&&nums[n-2]?ans:-1;
    }
};
```