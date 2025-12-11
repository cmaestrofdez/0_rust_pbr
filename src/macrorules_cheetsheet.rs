 
macro_rules! mr_0 {
    () => {println!("nada!")};
    ($a : expr)=>{println!("a! {:?}", $a)};
    ($a: tt + $b:tt)=>{println!("$a: tt + $ b:tt :{:?}", $a + $b)};
    ($a: ident $t:tt )=>{println!("b! {:?}", $a)};
}

macro_rules! mr_1 { 
    ($a:tt + $b:tt)=>{println!("$a: tt + $ b:tt {:?}", $a + $b)};
    
}
// se puede usar el atrib meta para hacer pattern matching de meta attributes tipo #[Debug] 

macro_rules! mr_2 { 
(#[no_mangle])=>{println!{"no manglee"}};
  (#[inline])=>{println!{"inline"}};
    
}
macro_rules! mr_3 {
    ($a:ident, $b:expr) => {
        let $a = 1;
        $a
    };
}
 
macro_rules! mr_4_recursive_each{
    ()=>{};
    ($_tt : tt $($rest:tt)*)=>{mr_4_recursive_each!($($rest)*);};
}
// There are two ways to expose a macro to a wider scope. The first is the #[macro_use] attribute. This can be applied to either modules or external crates. For example:
#[macro_use]
mod macros {
    macro_rules! X { () => { Y!(); } }
    macro_rules! Y { () => {} }
}
struct Recurrence{

    pub count_start : i64,
    pub count_end : i64 ,
    pub current_count : i64 ,

}
impl Recurrence{
    fn iter(&self)->RecurrenceIterator{
        RecurrenceIterator { count_start:0 , count_end:10, current_count:0 }
    }
}
struct RecurrenceIterator{
    pub count_start : i64,
    pub count_end : i64 ,
    pub current_count : i64 ,
}

impl Iterator for RecurrenceIterator {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        self.current_count+=1;
        if self.current_count>=self.count_end {
            return None
        }
        Some((self.current_count))
    }
     
    
}
macro_rules! create_instance {
    ( 0,...,3) => {
         [0,1,2,3]
    };
    ( a[n]=0,...,3) => {
        [0,1,2,3]
   };
   ( a[n]=$e:expr,...,3) => {
        [$e,1,2,3]
    };
    ( a[n]=$e:expr,...,a[o]=3) => {
        [$e,1,2,3]
    };
    ( a[n]=$e:expr,$($e0:expr)+,a[o]=3) => {
         [0,1,2,3]
    };
}
macro_rules! make_rec {
    (  ) => {  0 };
    ( $a : expr ) => { 1 };
    ( $a : expr , $($rest:expr),+) => {1 + make_rec!($($rest:expr),*)};
    
}
// https://danielkeep.github.io/tlborm/book/pat-callbacks.html
//https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/
// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-4/blob/master/CudaBVH.cpp
 //         https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-4/blob/master/renderkernel.cu
#[test]
pub fn examples() {
    let ads = "1";
    mr_0!(1);
    mr_0!(ads);
    mr_1!(1+1);
    mr_3!(a, 1);
 let a = create_instance!( 0,...,3);
 let a = create_instance!( a[n]=1,12, a[o]=3);
 let ads = "1aass";
 let rec = Recurrence{count_start:0 , count_end:10, current_count:0};
 for e in rec .iter(){
    println!("{:?}", e);
 };
 
}
 