
// cargo run RUSTFLAGS=-Awarnings


// #![allow(dead_code)]
// #![allow(unused_variables)]
//#![deny(warnings)]
mod main2;
use main2::modulo1::{modulo_fn};

use std::fs::File;
use std::io::Error;
            use std::io::Result;
            use std::io::ErrorKind;
            use std::path::Path ;

struct Person{n : i32} 
struct S{
    m : i32,
    a : f32,
    f : char,
    s :String,
    arrayfixed : [f32; 12],
    arrayfixed2 : [char; 12],
    // arrayfixed2 : [String; 12],
    vector : Vec<i128>,
}
fn write_img( a : &[u8], m:(i32,i32)){}
fn write_img_t<T>( a : &[T], m:(i32,i32)){}
fn write_img_3<T>( a : &mut  [T], m:&mut(i32,i32))->i32{
    1
}
fn write_img_4<T>( a : &mut  [T], m:&mut(i32,i32))->(i32, i32){
    (1, 1)
}
fn write_img_5<T>( a : T){}


struct Rec<T>{
    x:T, 
    y:T,
}
impl <T: std::cmp::PartialOrd + std::cmp::PartialEq>  Rec<T>{
    fn compare(&self)->bool{
        true
    }
    fn can_hold(&self, other : Rec<T>)->bool{
        other.x > self.x;
        true
    }
}

struct Guess<T>{
    t : T
}
impl<T> Guess<T>{
    pub fn new(t:T)->Guess<T>{
        panic!("TOTAL!");
        return Guess{t};
    }
    pub fn new1(t:i32)->Guess<i32>{
        if t>100{
            panic!( "less than or equal to 100")
        }else if  t < 100 {
            panic!( "less than or equal to 100" )
        }
        return Guess{t};
    }
}


#[cfg(test)]
mod tests{
 use super::*; //esto trae el modulo superio a dentro de este modulo
  #[test]
  fn it_works(){
    let rs = 2+1;
    assert_eq!(rs, 3);
    assert_ne!(rs,  13);
    assert!(true); // boolean
  }
  #[test]
  fn test_rect(){
    let r : Rec<f32> =  Rec{x:1.0,y:1.0};
    let r1 : Rec<f32> =  Rec{x:1.0,y:1.0};
    
    assert!(r.compare()); // boolean
    assert!(r.can_hold(r1)); 
    //podemos añadir un text
    // assert!(false, "esto es lo que debe salir {}",true);
    
    
  }
  #[test]
  #[should_panic]
  fn esto_debe_fallar(){
    // podemos comprobar que ha saltado una exception
    let g : Guess<f32> = Guess::new(1.);

  }
  
  #[test]
  #[should_panic(expected = "less than or equal to 100")]
  fn esto_debe_fallar_1(){
    // podemos comprobar que ha saltado una exception
    // cuando falla es 
    let g : Guess<i32> = Guess::<i32>::new1(1);

  }

//   /**
//    * podemos hacer los test selection de nombres
//    * usando un patterns
//    *    cargo test mi_test_que_hay_que_pasar
//      tb podemos usar un trozo
//         cargo test hay_que_pasar
//    */
  #[test]
  fn mi_test_que_hay_que_pasar_1(){
    assert!(true); // boolean
  }

  #[test]
  fn mi_test_que_hay_que_pasar_2(){
    assert!(true); // boolean
  }

  /**
   * ignore
   * si queremos pasar los test ignorados tenemos que hacer
   *    cargo test -- --ignored
   * 
   */
  #[test]
  #[ignore]
  fn mi_test_loooong(){
    assert!(true); // boolean
  }
}

/**
 * 
 *  modules
 */
mod mi_mod_parent{
    pub fn mi_meth(){}
    pub mod mi_child{
        use super::*;
        pub fn mi_child_method(){
            super::mi_meth();
        }
    }

    pub struct Num{
        pub n :i32
    }
    impl Num{
        pub fn new()->Num{
            Num{n:0}
        }
    }

    pub enum Things{
        No,Si
    }
    pub struct MiStruct {
        i:i32,

    }
    
    impl MiStruct{
        fn mi_impl(&self)->i32{
            21
        }
    }
    pub struct MiStruct1{

    }
    pub mod mi_child_1{
        pub struct MiStruct12{

        }
    }
    
    pub mod use_things{
        use crate::mi_mod_parent::MiStruct as TheNewNameForStruct;
        use crate::mi_mod_parent::{MiStruct, MiStruct1, mi_child_1::MiStruct12};
        pub fn main()->TheNewNameForStruct{
            let mut m : &MiStruct12 =  &MiStruct12{};
            TheNewNameForStruct{i:1}
        }
    }
     
    
    
    pub mod test_data{
        #[test]
        fn test_mi_mod(){
            use super::*;
            use crate::mi_mod_parent::Things::No;
            use crate::mi_mod_parent::MiStruct;                 
            use crate::mi_mod_parent::MiStruct as TheNewNameForStruct;                 
            assert!(true);
            MiStruct{i:1};
            TheNewNameForStruct{i:1};
        }
    }
}

// https://doc.rust-lang.org/book/ch12-01-accepting-command-line-arguments.html
// cargo G:\vs_codes\2_rust\hello\src\poem.txt
fn mi_parser(v : &[String])->(&str, &str){
    let query = &v[1];
    let filepath = &v[2];
    (query, filepath)
}
struct Config {
    filepath:String,
    query:String,
}
impl Config {
    fn new(v:&[String])->Config{
        let query :String = v[1].clone();
        let filepath: String=v[2].clone();
        Config{query,filepath}
    }
    fn build(v:&[String])->Result<Config >{
        let query :String = v[1].clone();
        let filepath: String=v[2].clone();
        if(v.len()>3){
             panic!("");
        }
        Ok(Config{query, filepath})
    }

}
fn mi_parser_config(v : &[String])->Config{
    Config::new(v)
}
fn mi_read(c:Config)->Result<String>{
    std::fs::read_to_string(c.filepath)
}
struct LineNum{i:i32}
fn search<'a>(q:&str, contents:&'a str)->Vec<&'a str>{
    let mut v = Vec::new();
    let mut vv = Vec::<LineNum>::new();

    for line in contents.lines(){
        if line.trim().contains(q) {
            v.push(line);
            vv.push(LineNum{i:1});

        }
    }
    vec![]
}

/**
 * 13
 * https://doc.rust-lang.org/book/ch13-01-closures.html
 */


 #[derive(Clone,Debug, PartialEq, Copy)]
enum MiShirtColor{
    Red, Blue
}
struct Inventary{
    shirts : Vec<MiShirtColor>
}
impl Inventary{
    fn giveaway(&self, user_pref  : Option<MiShirtColor>)->MiShirtColor{
        MiShirtColor::Blue
    }
    fn most_stocked(&self)->MiShirtColor{
        let mut numRed  = 0;
        let mut numBlue  = 0;
        for shirt in &self.shirts{
            match shirt{
                MiShirtColor::Blue=>numBlue+=1,
                MiShirtColor::Red=>numRed+=1,
            }
        }
        MiShirtColor::Blue
    }
     fn mi_opts(&self)->Option<MiShirtColor>{
        Some(MiShirtColor::Blue)
    }
    
}










fn main_closures(){
    let _clus  = |num:u32|->u32{
        num
    };
    fn mi_clores (x:i32)->i32{
        x
    }
    let mi_closure_with_ops = |_x:i32|->Option<i32>{
        Some(32)
    };
    let mitest:bool =  match mi_closure_with_ops(1){
        Some(32)=>false, 
        _=>true,
    };
    let res:i32 =  mi_closure_with_ops(2).unwrap();

    let inver : Inventary = Inventary{ shirts: vec![]};
    let r : MiShirtColor =  match inver.mi_opts() {
        Some(MiShirtColor::Blue)=>MiShirtColor::Blue,
        Some(MiShirtColor::Red)=>MiShirtColor::Red,
        None=>MiShirtColor::Red
    };
}
#[derive(Debug)]
struct Rect{
  w:i32,
}

fn main_closures_1(){
    let mut list =[Rect{w:12}, Rect{w:1}, Rect{w:13}];
    let mut opts : i32 = 0;
    list.sort_by_key(|r|{ 
        opts+=1;
        r.w });
    println!("{:#?} , {}",list, opts);
}
// main main_closures_1();


//** 13.2 Processing a Series of Items with Iterators */
fn  iter_1(){
    let  list : Vec::<Rect>  =vec![Rect{w:12}, Rect{w:1}, Rect{w:13}];
    let mut iter_list =  list.iter();
    for val in iter_list{
        print!("{}" ,val.w);
    }
    let v1 = vec![1, 2, 3];

    let mut v1_iter = v1.iter();
    
    let integ : &i32 =  match v1_iter.next(){
        Some(i)=>i,
        _=>&0
    };
}
fn  iter_2(){
    let v1 = vec![1, 2, 3];
    let mut v1_iter = v1.iter();
    let v2 : Vec::<i32> = v1_iter.map(|x| x +1 ).collect();
    let mut vstr : Vec::<i32> = Vec::<i32>::new();
    // let vstr_res : Vec::<i32> =  vstr.iter().filter(|n| n==1).collect();
}

struct Pooint{
    x: i32,
    y: i32
}
enum Mesn {
    Msn, 
    Msni32(i32),
    MsnOpt(Pooint),
    MsnOpt1((Pooint, Pooint)),
    MsnOpt2((Pooint, Pooint)),
}
trait Float{
    const ZERO:f32 =0.0;
}
impl Float for f32{
    const ZERO:f32 =0.0;
}
 
mod clousures_ex{
       
     
     use std::ops::Fn; 
     struct Cache<T> where T:Fn(i32)->i32{
        calculation : T,
        value : Option<i32>
     }
     impl <T>Cache<T>  where T:Fn(i32)->i32{
        fn new( calculation : T)->Cache<T>{
            Cache{
                calculation,
                value :None,
            }
        }
        fn value_(&mut self, arg : i32){
            match self.value{
                Some(v)=>v,
                None => {(self.calculation)(arg)},
            };
        }
     }


     pub fn mi_ex1(){
        let mi_closure = |num:i32|->i32{0};
        let mut c  = Cache::new(mi_closure);
        c.value_(1);

        let mut x : i32 = 1;
        let mi_move_clousere_sintax =  |num:i32|{
            x+num
        };

     }

     
     pub fn iter_again(){
        use std::iter;
        let v :Vec<i32>  = vec![1,2];
        let mut v_itr  = v.iter(); 
        let mut y : &i32 = v_itr.next().unwrap() ;

        for value  in v_itr{
            println!("{} ", value);
        }



        let otr :Vec<i32> = vec![1,2,3,4];
        let mut oth_iter   = otr.iter();
        let mut total : i32 =  oth_iter.sum();
        println!("{} ", total);


        let otr1 :Vec<i32> = vec![1,2,3,4];
        let mut otr1_ve : Vec<_> = otr.iter().map(|x : &i32| x +1 ).collect();

        
        
        println!("{:?} ", otr1_ve );

     }
     #[derive(PartialEq, PartialOrd)]
     struct Shoe{
        size: u32,
     }

     fn shoes_in_mi_size(shoes: Vec<Shoe> , shoe_size: u32 )->Vec<Shoe>{
       let m : Vec<_> =   shoes.into_iter().filter(|shoe| shoe.size == shoe_size).collect() ;
        m
     }

     //como hacer un iterator
     // 1. hacer la struct
     // 2. escribir el trato
     struct Counter {
        count : u32,
     }
     impl Counter{
        fn new()->Counter{
            Counter{count:0}
        }
     }

     // 2. hacemos el traity
     impl Iterator for Counter{
        type Item = u32; // esto sel llama typo asociado
        fn next(&mut self )->Option<Self::Item>{
            if self.count < 10 {
                self.count +=1;
                Some(self.count)
            }else{
                None
            }
            
        }
     }
     fn test_custom_iter(){
        let mut c =  Counter::new();
        let i0 : Option<u32> = c.next(); 
        
     }
     fn test_custom_iter_1(){
        let mut c =  Counter::new();
        let mm :Vec<_> =  c.map(|x| x).filter(|x : &u32 | x%3==0).collect();
        let res : u32 = mm.iter().sum::<u32>();
     }
     fn test_custom_iter_2(){
        let mut c =  Counter::new();
        let  res : Vec<_> = c.zip(Counter::new()).map(|(a, b )| a*b).filter( |x| x%3==0 ).collect();
     }

}

mod  smart_pointers_ex{
    use std::rc::Rc;
    struct CustomSmartPointer{
        data :String
    }
    impl Drop for CustomSmartPointer{
        fn drop(& mut self){
            println!("drop data {}", self.data);
        }
    }
    pub fn smart_ptrs(){
        let m  = Box::new(1);
        

        {
            CustomSmartPointer{data:String::from("esto es el dato")};
        }
        let mm = 1;
        drop(mm);
    }
    pub fn rc_impl(){
        let i = 1;
        let t = Rc::new(i);
        let t2 = Rc::clone(&t);
        let t3 = Rc::clone(&t);
        
        println!("{} ",Rc::strong_count(&t) );
    }
    pub fn main_smart_pointers_ex(){
        rc_impl();
        // smart_ptrs();
    }
}
mod ths{
    use std::thread;
    use std::time::Duration;
    pub fn main_ths(){
        let handler =  thread::spawn(||{
            for i in 1..10{
                println!("thread_spawn -> {}", i);
                thread::sleep(Duration::from_millis(122));
            }
        });
        handler.join().unwrap();
        thread::sleep(Duration::from_secs(1));
        println!("thread_main end! " );
    }
    pub fn main_ths_1(){
        let mut v = vec![1,2];
        let handler = thread::spawn(move ||{
            v[1] = 1;    
        });
        
        handler.join().unwrap();
        
    }
    use std::sync::mpsc;
    pub fn mpsc_main(){
        let (t, r) = mpsc::channel();
        let handler = thread::spawn(move ||{
            let msn = String::from("esto es lo que yo mando");
            t.send(msn).unwrap();
        });
        let rev_string  = r.recv().unwrap();
        println!("main_recv{} ", rev_string);

    }
    pub fn mpsc_main_2(){
        let (t, r) = mpsc::channel();
        thread::spawn(move||{
             let v = vec![
                String::from("uno"), 
                String::from("dos"),
                String::from("tres")

             ];
              for cad in v {
                t.send(cad).unwrap();
              }
        });
        for reception in r{
            println!("{}", reception);
            thread::sleep(Duration::from_secs(1));
        }
    }
    pub fn mpsc_main_3(){
        let (t, r) = mpsc::channel();
        let tx2 = t.clone();
        thread::spawn(move||{
             let v = vec![
                String::from("uno"), 
                String::from("dos"),
                String::from("tres")

             ];
              for cad in v {
                t.send(cad).unwrap();
              }
        });
        thread::spawn(move||{
            let v = vec![
               String::from("tx2::uno"), 
               String::from("tx2::dos"),
               String::from("tx2::tres")

            ];
             for cad in v {
               tx2.send(cad).unwrap();
             }
       });
        for reception in r{
            
            println!("{}", reception);
            thread::sleep(Duration::from_secs(1));
        }
    }
    pub fn mpsc_main_shared_mem_4(){
        println!("mpsc_main_shared_mem_4");
        use std::fs;
        use std::fs::File;
        use std::io;
        let (sender, receiver) = mpsc::channel();
        let docus = vec![String::from("hello.txt")];
        let handler=  thread::spawn(move ||{
            
            
            for filename in docus{
                let mut f = File::open(filename).expect("msg: &str");
            
                let mut txt = String::new();
                std::fs::read_to_string(& mut txt);
                println!("{} ",txt);
                if sender.send(txt).is_err(){
                    break;
                }
            
            }

        });
        for rcv in receiver {
            println!("{}",rcv);
        }

    }


}

// https://www.youtube.com/watch?v=ReBmm0eJg6g&list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8&index=32

mod trait_objetcs_in_rust{
    pub trait Draw{
        fn draw(&self);

    }
    pub struct Screen{
        components : Vec<Box<dyn Draw>>,
    }
    pub struct Button{}
    impl Draw for Button{
        fn draw(&self){}
    }
    impl Screen {
        pub fn run(&self){
            for component in self.components.iter(){
                    component.draw();
            }
        }
    }
    pub fn trait_objetcs_in_rust_main(){
        let sc = Screen{components:vec![
                Box::new(Button{})
            ]
        };
        for element in sc.components.iter(){

        }
    }

    pub mod programing_rust_trais{
        pub struct Canvas{}
        impl Canvas {
            pub fn min_fn(&self)->(i32, i32){
                (1,1)
            }
            pub fn create_range(&self,x:i32)->std::ops::Range<i32>{
                x-1..10
            }
        }
        pub trait Visible{
            fn draw(&self, canvas : &mut Canvas);
            fn draw1(&self);
        }
        impl Visible for Canvas{
            fn draw(&self, canvas : &mut Canvas){
                canvas.min_fn();    
            }
            fn draw1(&self){
                let r = self.create_range(12);
                
            }
        }
        pub trait Visible1defaultImpl{
            
            fn draw_impl(&self)->std::io::Result<(usize)>{
                Ok(1)
            }
        }
        impl Visible1defaultImpl for Canvas{
            fn draw_impl(&self)->std::io::Result<(usize)>{
                Ok(1)
            }
        }

        pub fn main_programing_rust_trais(){
            let c : Canvas =Canvas {};
            let c1 : Canvas =Canvas {};
            let o = c1.draw_impl().unwrap();
            println!("{}",o);
            let x = match c1.draw_impl(){
                Ok(x)=>x,
                Err(e) => panic!("Problem creating the file: {:?}", e),
            };
            
            println!("{}",x);
            
            
        }
        // el Self como tipo retornado
        // no es compatible con un object trait
        pub trait Spliceable{
            fn spice(&self)->Self ;
        }
        struct CherryTree{}
        struct Mammoth{}
        impl Spliceable for CherryTree{
            fn spice(&self)->Self{
                CherryTree{}
            }
        }
        impl Spliceable for Mammoth {
            fn  spice(&self) ->Self{
                Mammoth{}
            }
        }
        // subtratos 
        pub trait Visible1 {
            fn positionVisible(&self)->(i32) {(1)}
        }
        pub trait Creature : Visible1{
            fn position(&self)->(i32) {(1)}
        }
        // cada tipo que implementa creatura tien que implementar Visible
        pub struct Broom;
        
        impl Visible1 for Broom {
        }
        impl Creature for Broom {
        }
        pub fn mi_metodo(m : Broom){
            m.position();
            m.positionVisible();
            
        }

        pub fn subtrato(){
            let b :Broom= Broom{};
            b.position();
            b.positionVisible();
            mi_metodo(b);
            

        }
        
        //type associate functions
        pub trait StdStringSet{
            fn new()->Self;
            // fn from_slice(string : &[&str])->Self;
            fn contains(string : &[&str])->bool;
            fn printInfo(){}
        }
        struct SortedString {}
        impl  StdStringSet for SortedString  {
            fn printInfo(){}
            fn contains(string : &[&str])->bool{
                true
            }
            fn new()->Self{
                SortedString{}
            }
        }
        pub fn typeAssociateFunctions(){
            let sorted = SortedString{};
            SortedString::printInfo();
            // SortedString::contains(String::from(""));
        }
        pub fn unknownsWords<S:StdStringSet>(doc : & [String])->S{
            S::printInfo();
            // retorna esto! 
            S::new()
        }
        
        //tipo asociado e iter 226
        pub fn collect_iter<I:Iterator>(iter :  I)->Vec<I::Item>{
            let newv = Vec::<I::Item>::new();
            let newv  = iter.collect();
            newv
        }
        pub fn iter_content_print<I>(iter : I) where I : Iterator , I::Item: std::fmt::Debug{
             for (index, item) in iter.enumerate(){
                println!("{},{:?}", index, item );
             }
        }
        pub fn iter_content_print_1(iter : &mut dyn Iterator<Item=String> ){}
        pub fn iter_content_print_2(iter : &mut dyn Iterator<Item=i32> ){}
        pub fn main_collect_iter(){
            let mut v = vec![1,2,3];
            let v1 = collect_iter(v.iter());

            let mut iv = vec![1,2,3];
            
        }
        // sdobre carga de operadores
        // const en tratos
        trait Float {
            const FLOAT_ONE : Self;
            // const FLOAT_MIN : Self;
            // const FLOAT_ONE : Self;
            // const FLOAT_ZERO : Self;
        }
        impl Float for f32{
            const FLOAT_ONE :f32   = 1.0;
        }
        impl Float for f64{
            const FLOAT_ONE :f64   = 1.0;
        }
        
        fn add_one<T:Float + std::ops::Add<Output = T>>(value : T)->T{
            T::FLOAT_ONE + value
        }

        pub fn main_associated_const(){
            add_one(1.0);
        }

        // 
    }

}

mod overload_operators{
    // pag 288
    struct Complex<T>{
        r : T, 
        i : T
    }
    use std::ops::Add;
   //1 // esto solo es para i32
    // impl Add for Complex<i32>{
    //     type Output = Complex<i32>;
    //     fn add(self, rhs : Self)->Self{
    //         Complex{
    //            r:  self.r+rhs.r,
    //            i:  self.i+rhs.i
    //         }
    //     }
    // }

    // algo mas generico, para todos los tipos que tengan el trait Add con el mismno tipo
   //2 // impl<T> Add for Complex<T> where T : Add<Output=T>{
    //     type Output = Self;
    //     fn add(self, rhs:Self)->Self{
    //         Complex{
    //             r:  self.r+rhs.r,
    //             i:  self.i+rhs.i
    //         }
    //     }
    // }
    impl<L, R> Add<Complex<R>> for Complex<L> where L : Add<R>{
        type Output = Complex<L::Output>;
        fn add(self, rhs : Complex<R>)->Self::Output{
            Complex{
                r:  self.r+rhs.r,
                i:  self.i+rhs.i
            }
        }
    }

    // unary ops
    // trait neg y not
    use std::ops::{Neg, Not};
    impl <T> Neg  for Complex<T>where T : Neg<Output=T>{
            type Output =Complex<T>  ;
            fn neg(self )-> Self::Output  {
                Complex{r:-self.r , i:-self.i
            }
        }
    }
    

    // operator y asignacion AddAssign;
    use std::ops::AddAssign;
    impl<T>  AddAssign for Complex<T> where T : AddAssign<T>{
        fn add_assign(&mut self, rhs: Complex<T> ){
            self.i+= rhs.i;
            self.r+= rhs.r;
         
        }
    }
    

    pub fn overload_operators(){
        let c  = Complex{r:0,i:0};
        let c1  = Complex{r:0.0,i:0.0};
        let c2 = Complex{r:0.0,i:0.0};
        let c3 = -c2;
        let mut csum = Complex{r:0.0,i:0.0};
        let step = Complex{r:0.0,i:0.0};
        csum +=step; 
    }

    // a[i] 321.pdf
    struct  Image<P>{
        w : usize,
        pxs : Vec<P>,
    }
    impl <P:Copy+Default> Image<P>{
        fn new(w:usize)->Image<P>{
            Image{
                w,
                pxs:vec![P::default();w]
            }

        }
    }

    impl<P> std::ops::Index<usize> for Image<P>{
        type Output = [P];
        fn index(&self, row : usize)->&[P]{
            &self.pxs[0..row]
        }
    }
    impl<P> std::ops::IndexMut<usize> for Image<P>{
        
        fn index_mut(&mut self, row : usize)->&mut [P]{
            &mut self.pxs[0..row]
        }
    }
    


}
mod utility_traits_ch_13_323{
    struct MiCosa{
         x :i32,
         // y:Vec<String>,
    }
    impl Drop for MiCosa{
        fn drop(&mut self){
            self.x=0;
            // self.y=Vec<String>::new();
        }
    }
    struct Selector<T>{
        elements : Vec<T>,
        current :usize
    }
    use std::ops::{Deref};
    impl <T> Deref for Selector<T>{
        type Target = T;
        fn deref(&self)->&T{
            &self.elements[self.current]
        }
    }
    fn main_test_deref(){
        let s =  Selector{elements:vec![1,23],current:1};
        let mut h = *s;
        println!("{}",h);
        
    }
    use std::default::Default ;
    //default
    struct MiStructDef<T :  std::default::Default>{
        cad : String ,
        n : T
    }
    
    impl <T:Default> Default for MiStructDef<T>{
        fn default()->MiStructDef<T>{
            MiStructDef{cad  : String::from(""), n:T::default()}
        }
    }
    //este es muy util:
    // transform o hace un cast implicito sobre el parametro 
    // del arg llamado al trato Into, o From
    // si queremos que implicitamente lo haga con un objeto nuestro 
    // tenemos que añadir el trato en el objecto
    use std::convert::{Into, From};
    // impl <T> Into for MiStructDef{}
    fn ping<A>(addr:A) where A : Into<std::net::Ipv4Addr>{
         let ipv4 :std::net::Ipv4Addr =  addr.into();
    }


}
mod closures_14{
    // recuerdo : las closure son structs con el constructor como parametro
    // y el operador sobre escrito!
    // Fn,  se llama multiples veces
    // Fn es una funcion 'singleton'
    fn std_form<F>(f:F) where F :Fn(i32)->bool{}
    fn std_form_fnonce<F>(i:i32,f:F) where F :FnOnce(i32)->bool{
        f(i); 
    }
    fn add_closure(){
        let mut v = vec![];
        let f =  |mut otr:Vec<i32>|{
            otr.push(1);
        };
        f(v);

    }
    pub fn  main_closures(){
        std_form(|x:i32|{true});
        let fnunavez = |x:i32|{true};
        std_form_fnonce(0,fnunavez);
        let withoutargs = ||{true};
    }

    use std::collections::HashMap;
     // crear un router
     struct Request{}
     struct Response{}

     type BoxedCallBack = Box<dyn Fn(&Request)->Response>;
     struct BasicRouter{
        router : HashMap<String, BoxedCallBack>
     }
     
     
     impl BasicRouter {
        pub fn new()->BasicRouter{
            BasicRouter{router:HashMap::new()}
        }
        // añadimos un object trait usando el dyn Fn y metiendolo en el heap(creo)
        fn add_router(& mut self, route : String, cb : BoxedCallBack){
            self.router.insert(route,cb);
        }
        fn handle_request(&self, url :String){
            let r = Request{};
            match self.router.get(&url){
                None=>Response{},
                Some(content_callback)=>content_callback(&r)
            };
        }
     }
     pub fn mi_basic_router(){}


     // function pointers 364
     struct BasicRouter1{
        // esto se llama fn pointer... representa eso, un fun pointer
        routes : HashMap<String, fn(&Request)->Response>
     }
     impl BasicRouter1{
        fn new()->BasicRouter1{
            BasicRouter1{
                routes:HashMap::new()
            }
        }
        //aqui pasamos el la fn ... usando la keywoed fn
        fn add_route( & mut  self, request : Request, response : fn(&Request)->Response){
            self.routes.insert(String::from(""), response);
        }
     }


     
     pub fn function_pointers_364(){
        fn mi_fn_ptr(i:i32)->bool{true}
        let mi_fn : fn(i32)->bool =|i:i32|{true};
        let mi_fn2 : fn(i32)->bool =mi_fn_ptr;



     }

}
mod iterators_15{
    pub fn mi_iteraror(){
        let mut v  = vec![12,3,4];
        for eleme in &v{};
        for eleme in & mut v{};
        for eleme in v{};
        
        let l : Vec<i32> = std::iter::from_fn(||{Some(0)}).take(100).collect();
    }
    pub fn mi_iteraror_1(){
        
        let  txt = "esto \nse".to_string();
        
        let r  : Vec<&str> =  String::from("asdad").lines().filter(|s| *s != "se").collect();
    }
    pub fn iter_filter_map(){
        use std::str::FromStr;
        
        let  txt = "esto \nse 12".to_string();
        for tx in txt.split_whitespace().filter_map(|w| f64::from_str(w).ok()) {
            println!("{:4.2}",tx);
        }
        

    }
    pub fn iter_flat_map(){
        use std::collections::HashMap;
        let mut countries_values = HashMap::new();
        countries_values.insert("japan", vec!["blabla"] );
        countries_values.insert("japan", vec!["blabla"] );
        countries_values.insert("japan", vec!["blabla"] );
        let countries  =  vec!["japan"];
        for element_from_v in countries.iter().flat_map(|element|  &countries_values[element] ){
            println!("{}", element_from_v);
        }
        
        // flatte un vector con options
        let mut mv = vec![None,Some(""), None];
        mv.into_iter().flatten();
    }
    pub fn take_while_ejemplo(){
        // take_lines() conserva el estado de modo que itera hasta que alguna condificion (predicado sea) false
        let message = "To: jimb\r\n\
                       From: superego <editor@oreilly.com>\r\n\
                       \r\n\
                       Did you get any writing done today?\r\n\
                       When will you stop wasting time plotting fractals?\r\n";
        for res in    message.lines().take_while(|lines| !lines.is_empty()){
            println!("{}",res)
        }
    }
    pub fn skip_ejemplo(){
        // take_lines() conserva el estado de modo que itera hasta que alguna condificion (predicado sea) false
        let message = "To: jimb\r\n\
                       From: superego <editor@oreilly.com>\r\n\
                       \r\n\
                       Did you get any writing done today?\r\n\
                       When will you stop wasting time plotting fractals?\r\n";
        for res in    message.lines().skip(2){
            println!("{}",res)
        }
        for res in    message.lines().skip_while(|line| !line.is_empty()){
            println!("{}",res)
        }



    }
    pub fn reverse_ejemplo(){
        let v= vec!["uno", "dos", "tres"];
        let it =  v.iter(); 
        // assert_eq!(it.next_back(), Some("tres"));
        // assert_eq!(it.next(), Some("uno"));
        // assert_eq!(it.next(), Some("dos"));
    }
    pub fn chain_ejemplo(){
        let v= vec!["uno", "dos", "tres"];
        let v1= vec!["4", "5", "6"];
        let it = v.iter();
        let it1 = v1.iter();
        let itre  =it.chain(it1);
        
        for item in itre.rev() {
            println!("{}", item);
        }
      
    }
    pub fn main_iterator_max(){}
    pub fn main_iterator_15(){
        // chain_ejemplo();
        // iter_flat_map();
        // take_while_ejemplo();
        // skip_ejemplo();
    }

    //
    
 }
 mod collections_16{
    pub fn vector_collections_16(){
        {
            let mut va =vec![1,2,3];
        let vb =vec![1,2,3];
        va.extend(vb.iter());
        }
        {
            let mut vc :Vec<i32>=(1..10).collect();
            let vd :Vec<i32> = vc.split_off(2);
            assert_eq!(vec![9,10], vd);
        }
        {
            let mut va :Vec<i32>=(1..10).collect();
            let mut vb :Vec<i32>=(1..10).collect();
             //va.append(vb);
        }
        {
            let mut va :Vec<i32>=(1..10).collect();
          
           va.retain(|x| *x >1 );
        }
        {
            assert_eq!([[1,2],[3,4]].concat(), vec![1,2,3,4]);
            assert_eq!([[1,2],[3,4],[3,4]].join(&0), vec![1,2,0,3,4,0,3,4]);
        }
        {
            // slices 
            let mut v = vec![1,2];
            let mut vsli  = (1..3).collect::<Vec<i32>>();
            let mut slic = [1,2,3,4];
            slic.first();
            slic.last();
            slic.get(0);
            slic.to_vec();
            slic.len();
            slic.is_empty();

            
            // last first
            if let Some(item )  = vsli.first(){
                item;
            }
            // get 
            
            let mut v = Vec::with_capacity(12);
            v.capacity();
            v.reserve(12);
            v.shrink_to_fit(); // Tries to free up the extra memory if vec.capacity() is greater than vec.len()
            // add 1 to end of vec
            v.push(1);
            // remove and return last element from vec
            v.pop();
            v.insert(1, 1);
            v.clear();
            // Like vec.truncate(index), except that it returns a Vec<T> containing the values
            // removed from the end of vec. It’s like a multivalue version of .pop().
            let index =1;
            v.split_off(index);
             let mut bb  : Vec<i32>=  Vec::with_capacity(12);
            //v.append( bb );
            // range 
            v.drain(1..3);

            //join
            
        }
        {
            use std::collections::HashSet;
            let mut s  =  HashSet::new();
            s.iter();
            s.insert("eso");
            s.insert("eso lo otro");
            s.replace("eso lo otro").unwrap();
            for e in s {
                println!("{}", e);
            }
        
            // #[derive(Debug,Hash,)]
            // struct mi_element<T>{
            //     i : i32, 
            //     n : String,
            //     t : T 
            // }
            // impl<T> PartialEq for mi_element<T>{
            //     fn eq(&self, other: &mi_element<T>)->bool{
            //         self.i == other.i
            //     }
            // }
            // impl <T> Eq for mi_element<T>{}
            // // impl<T> Clone for mi_element<T>{
            // //     fn clone(& self)->Self{
            // //             // *self
                        
            // //     }
            // // }
            // impl<T> mi_element<T>{
            //     pub fn new( i : i32,  t : T)->mi_element<T>{
            //         mi_element{i, n : String::from(""), t }
            //     }
            //     pub fn mi_method(self)->i32{
            //         self.i
            //     }
            //     pub fn mi_method1(self)->Option<i32>{
            //         Some(self.i)
            //     }
            // }
            // let mmmm = mi_element::new(1,2);
            // mmmm.mi_method();
            // if let Some(i) = mmmm.mi_method1(){
            //     println!("{}", i);
            // }
            
            let mut m =  HashSet::new();
            m.insert(1);
            m.insert(2);
            let mut m2 =  HashSet::new();
            m2.insert(3);
            m2.insert(4);
            let mv : Vec<_> = m2.union(&m).collect();

            let mut setele =   HashSet::<i32>::new();
            // setele.insert(mi_element::new(1,1));
            // for e in setele {
            //     e.i;
            //     e.n;
            // }

            
            // m.insert(mi_element{i:2});
            // m.insert(mi_element{i:3});
            // m.intersection(other: &'a HashSet<T, S>)


            
            // s.get(value: &Q)
            // s.take(value: &Q)
            
        }

    }
    pub fn main_collections_16(){
        
    }
 }
 mod string_And_text_chap17{

    pub fn main_string_And_text_chap17(){
        use std::path::*;
        use std::fs::DirEntry;
        use std::fs::ReadDir;
        
        let p  = Path::new(".");
        for r in  std::fs::read_dir(p) {
            
        }
        p.read_dir().expect("msg: &str").map(|dent| dent.unwrap().path()).map(|ff|print!("--{:?}",ff.is_file()));
        let d  =    std::fs::read_dir(Path::new(".")).expect("msg: &str");
    
        if p.is_dir(){
            
            
            let entrydur =   p.read_dir().unwrap();
            for d in entrydur{
               let dd = d.unwrap();

               println!("{:?}",dd.path());
               
               
            //    if newpath.is_dir(){
            //         let a =   std::fs::read_dir(newpath).unwrap();
            //          for aa in a {
            //              if aa.is_ok(){

            //              }
            //          }
            //    }

               
            }
            
        }
        
        
    }
    pub fn main_string_And_text_chap17_1(){
        let paths = std::fs::read_dir( "./").expect("problems");
        for p  in paths {
            // p.unwrap().metadata().modified()?;
            println!("{}",p.unwrap().path().display());
            ;
        }
        use std::env;
         let r = env::current_dir().expect("ups");
         let m = std::fs::read_dir(r).unwrap();
         for p in m  {
            println!("{}",  p.unwrap().path().display());
        }

        use std::path::PathBuf;
        let  pathSrc = PathBuf::from("./src");
        std::fs::read_dir(pathSrc).unwrap();
         let rdm = std::fs::read_dir( PathBuf::from("./src") ) .unwrap();
         for r in rdm  {
         //   println!(" {}", r.unwrap().path().display());
            let isO = std::env::current_dir().unwrap().join( r.unwrap().path() );
            // println!("-->{}",std::env::current_dir().unwrap().join( r.unwrap().path() ).is_dir());
            if  ! isO.is_file() && !isO.is_dir(){
                
                let r0 = std::fs::read_dir( isO ) . unwrap();
                for r_c in r0  {
                     
                }
            }
         }
    }

 }// mod string_And_text_chap17



fn main_test() {
    main2::rust_programing_cookbook::main();
    //main2::read_write_buffers::mi_main();
    return;
    main2::error_handling_chap7::structures_ch9::mi_fn_1();
   
    main2::error_handling_chap7::mainfn();
    main2::modulo1::modulo_fn();
    return;
    string_And_text_chap17::main_string_And_text_chap17_1();
  //  string_And_text_chap17::main_string_And_text_chap17();
    return;
    iterators_15::main_iterator_15();
    return;
    trait_objetcs_in_rust::programing_rust_trais::main_programing_rust_trais();
    return;
    let uno = Some(5);
    let dos = Some(1);
    match( uno, dos){
        (Some(1),Some(1))=>true,
        (Some(_),Some(1))=>true,
        (Some(_),Some(5))=>true,
        (Some(_),Some(_))=>true,
        (_,_)=>false
    };
    let x  = Mesn::Msn;
    match x{
        Mesn::Msni32(_y)=>print!(""),
        Mesn::MsnOpt(_p)=>print!(""),
        Mesn::MsnOpt1((_a, _b))=>print!(""),
        Mesn::MsnOpt2((Pooint{x:_b, y}, Pooint{x:a,y:yy}))=>print!(""),
        _=>print!("nada")
    };
    let p  : Option<&str> =None;
    
    if let Some(_y) = p{

    }

    
    // ths::mpsc_main_shared_mem_4();
    //smart_pointers_ex::main_smart_pointers_ex();
    //let mi_cl = |num:String|->String{ String::from("")};
//     clousures_ex::iter_again();

    // let v :  Vec<String> =  std::env::args().collect();
    // let query = &v[1];
    // let filepath = &v[2];
    // print!("->{}\n",v[1]);
    // print!("filepath->:{}\n",v[2]);
    // let textcontent : String = std::fs::read_to_string(filepath).expect("debe haber un archivo llamado poem.txt");
    // print!("entire content of string : {}\n " , textcontent);
    // let cong  : Config = Config::build(&v).unwrap();

}

fn main1() {
    




    mi_mod_parent::mi_child::mi_child_method();
    mi_mod_parent::use_things::main();
    write_img_5(10.0);
    
    let v =  19_i8 as u16;
    let va =  19_i8 as u32;
    
    println!("{} {} {} {} {} {}",
            v, 
            va, 
            1_f32.powf(1_f32), 
            -1.12_f32,
            -1.12_f32.abs(),
            -1.12_f32

        );

    // i32::abs(1);
    // f32::sqrt(1.);
    // (false as i32)==0;
    // (false as i32)==1;

    // let c : char = 'c';
    // let singleq : char = '\n';
    // let mut  ii : i32 =  (singleq as  i32);
    /*
 assert_eq!('*'.is_alphabetic(), false);
assert_eq!('β'.is_alphabetic(), true);
assert_eq!('8'.to_digit(10), Some(8));
assert_eq!('޵.'len_utf8(), 3);
assert_eq!(std::char::from_digit(2, 10), Some('2'));
     
     */
 


     // tuples 
     let t = (1,2);
     let (a, b) = t;
     let (aa, _) = t;
     t.0;
     t.1;



     //
     // arrays
     let l : [u32; 2]=[1,2];
     let l2  =[1,2];
     let mut v0  = vec![1,2];
     v0.push(1);
        
     fn mi_new(b:(usize, usize)){
         let (x, y) = b;
         vec![0;y*x];
     }

     mi_new((1,1));
     let vite :  Vec<i32> = (0..4).collect();

     let mut bb :Vec<i32>  = Vec::new();

    bb.capacity();
    bb.len();

    bb = (0..32).collect::<Vec<i32>>();
    let mut v : (i32, i32) = (1,2);
    v.0 = 1;
    
    let mut p : Person =  Person{n:112};
    
    
    let num  = match  p.n  {
        1 => "ok",
        _=>"no ok"
    };
    use std::io;
    
    // expresiones
    let expr_res  = if 1 >10 { "ok"  } else {"no ok"};
    let block_sem = {
        "ok!"
    };
    let blck_otro = {
        let ii = 1 + 1;
        "ook!";
    };

    let mut per : Person = Person{n:1};
    //declaration
     let name;
     let mut name_;
     if(per.n == 1){
        name = "ADS";
     }

     name_ = match per.n {
        1 => "ok",
        _ => "nop"
     };
      

     let v : Vec<i32> = vec![1,23];
     for i in v {
        println!("{} ", i);
     }
     {
        fn mi_test( n : i32){
            true;
        }

        let vite :  Vec<i32> ;
        let mut resvite :  Vec<bool>  = vec![];
        vite =  (1..11).collect();
        for ele   in vite {
            let b : bool =   match ele{
                    0=> true,
                    1=> true,
                    _=> true,
                    // Some => true,
                    // None => true,

            };
            resvite.push(b);
        }

     }
     {

        fn route ( _i : Ip){}

        enum Ip {
            V2,
            V3
        }
        let ip0 : Ip = Ip::V2;
        route(ip0);

        struct IpAddr{
            ip : Ip  ,
            addr : String,
        }
        let _m : IpAddr =  IpAddr{
            ip : Ip::V2,
            addr: String::from("aa"),
         };




         enum Msn{
            Quit, 
            Move(i32,i32),
            Write(String),
         };
         struct Msnt{
            msn : Msn,
         }
         



     }
     // options
     {
        let x : i8 = 1;
        let y : Option<i8> = Some(5);
        let z : Option<i8> = None ;
        let w : i8  =  x + y.unwrap_or(0);
        let w0 : i8  =  x + z.unwrap_or(0);

        enum Coin{
            UNO, 
            DOS
        };
        fn choice(coin : Coin)->i32{
            match coin{
                Coin::UNO =>0, 
                Coin::DOS =>1
            }
        }
        
     }
     {

        fn choice1(x : Option<i32>)->Option<i32>{
            match x{
                None=>None,
                Some(i)=> Some(i+1)
            }
        }

        {
            let x0 : Option<i32> = Some(0);
            let x1 : Option<i32> = Some(1);
            let x2 : Option<i32> = None;
        }
        {
            let val : Option<i32> = None;
           if let Some(3) = val  {
                 println!("THREe")
           }
        }

        // Rust's Module System Explained!

        // Common Collections in Rust
        {
                // let a : [i32;_] = [];
                // let v : Vec<i32> =  Vec ::new (); // solo hay alloc. no ha inicializacion
                // let vv : Vec<i32> = vec![12, 12]; // alloc + inititializtion
                // let d : &i32 =  &vv[2];
                // // match vv.get(1){
                // //     Some(value)=> value,
                // //     None => 1,
                // // }
                // match vv.get(111){}
                

                // https://www.youtube.com/watch?v=Zs-pS-egQSs&list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8&index=8
        }
        {
            enum RowEnum {
                Int(i32),
                Txt(String)

            }
            let row :Vec<RowEnum> = vec![
                RowEnum::Int(1),
                RowEnum::Int(2),
                RowEnum::Txt(String::from(""))
            ];
            match &row[1] {
                RowEnum::Int(value)=>print!(" integet {} ",value),
                _=>print!("not a integet")
            }

        }

        {
            // // strings
            // https://youtu.be/Zs-pS-egQSs?list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8&t=648
            let s0 = String::from("eestp");
            let s0_1= String::from("eestp ");
            let s1 : &str = "bla";
            let s2 : String = s1.to_ascii_lowercase();
            let  s3 : String =  s0 + &s0_1 + "asd" + "asdsad" + s1 + &s2;
        }

        {
            let hello : String = String::from("eestp");
            let hello_ : &str =";";
            
        }

        // collections
        {
            use std::collections::HashMap;
            let mut hashmap : HashMap<String, i32> = HashMap::new();
            
            hashmap.insert(String::from("entrada"), 2);
            match hashmap.get(&String::from("otra"))   {
                _=>(),
            };
           let m : Option<&i32> =  hashmap.get("");

            for (a,  b) in &hashmap {
                
                println!("\n {} {}", a, b);
            }

            hashmap.entry(String::from("ba")).or_insert(1);
        }

        //handle error
        {
            // use std::fs::File;
            // use std::io::Result;
            // use std::io::ErrorKind;
            // use std::path::Path ;
            // https://doc.rust-lang.org/rust-by-example/std_misc/file/open.html
            let path  =  Path::new("");

            let fd : Result<File>  = File::open(&String::from(""));
            // match fd {
            //     Err(why)=>match why.kind() {
            //         ErrorKind::NotFound => print!("ups"),
            //         _=>print!("")
            //     },
            //     Ok(file)=>file,
            // }

            let greeting_file_result = File::open("hello.txt");


            let greeting_file = match greeting_file_result {
                Ok(file) => file,
                Err(error) => match error.kind() {
                    ErrorKind::NotFound => match File::create("hello.txt") {
                        Ok(fc) => fc,
                        Err(e) => panic!("Problem creating the file: {:?}", e),
                    },
                    other_error => {
                        panic!("Problem opening the file: {:?}", other_error);
                    }
                },
            };
            // https://doc.rust-lang.org/rust-by-example/std_misc/file/create.html
            // https://youtu.be/wM6o70NAWUI?list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8&t=452

            
        }
        {

          //  let greeting_file = File::open("hello_1.txt").unwrap();
           // let greeting_file_1 = File::open("hello_1.txt").expect("hello_1.txt debe estar en este proj");
            

        }
        {

                // fn read_username_from_file()->Result<String, io::Error>{
                //     let f  =  File::open("hello3.txt");
                //     let mut filed =  match f {
                //         Ok(file)=>file,
                //         Err(e)=> return Err(e)
                //     };
                //     let mut ret = String::new();

                //     match filed.read_to_string(&mut ret){
                //         Ok(_)=>Ok(ret),
                //         Err(e)=>Err(e),

                //     };

                // }



                use std::fs::File;
                use std::io::Error;
                            use std::io::Result;
                            use std::io::ErrorKind;
                            use std::path::Path ;
                            use std::io::{self, Read};
                

                            fn read_username_from_file() -> Result<String> {
                                let username_file_result = File::open("hello.txt");
                            
                                let mut username_file = match username_file_result {
                                    Ok(file) => file,
                                    Err(e) => return Err(e),
                                };
                            
                                let mut username = String::new();
                            
                                match username_file.read_to_string(&mut username) {
                                    Ok(_) => Ok(username),
                                    Err(e) => Err(e),
                                }
                            }

                            match read_username_from_file(){
                                Ok(data)=>print!("{}",data.as_str()),
                                Err(err)=>panic!("{}", err)

                            };

                            
                            /**
                             * The ? placed after a Result value is defined to work in almost the same way as the match expressions we defined to handle the Result values in Listing 9-6. 
                             * If the value of the Result is an Ok, the value inside the Ok will get returned from this expression, 
                             * and the program will continue. If the value is an Err, 
                             * the Err will be returned from the whole function as if we had used the return keyword so the error value gets propagated to the calling code.
                             */
                            fn read_username_from_file_con_inter()->Result<String>{
                                let mut username_file = File::open("hello.txt")?;
                                let mut username = String::new();
                                username_file.read_to_string(&mut username)?;
                                Result::Ok(username)
                            }
                            fn read_username_from_file_2()->Result<String>{
                                std::fs::remove_dir("dir");
                                std::fs::create_dir("src/dir");
                                
                                std::fs::read_to_string("hello.txt")
                            };
                            // read_username_from_file_2();
                             // ttps://youtu.be/wM6o70NAWUI?list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8&t=517
                            
                             // chap 10
                            {
                                fn largest<T>(list:&[T])-> &T{
                                    let mut m = &list[0];
                                    m
                                }
                                let mut m = vec![1,2,3];
                                largest(&m);
                                let mut mc = vec!['1'];
                                // largest(&mc);


                                struct m_st<T>{
                                    i : T,
                                    j:T,
                                };
                                struct m_st_3<T, U>{
                                    i : T,
                                    j: U,
                                };
                                let mut ms: m_st<f32> = m_st{
                                    i:1.1,
                                    j : 1.,
                                };

                                enum Option2<T>{
                                    M(T),
                                };

                                // methoid resolutions
                                struct Pt <T>{
                                    x:T,
                                    
                                }
                                impl <T> Pt<T>{
                                    fn x(&self)->&T{
                                        &self.x
                                    }
                                }
                                let mpt_32 : Pt<i32> =  Pt{x:1};
                                mpt_32.x();




                                struct Pt2<U,V>{
                                    x: U, y :V
                                };
                                // impl<U,V> Pt2<U,V>{
                                //     fn mixup<U, K>(&self, other : Pt2<U, K>)->Pt2<U,V>{
                                //         Pt2{
                                //             x : other.x,
                                //             y :  self.y,
                                //         }
                                //     }
                                // }












                                
                            }

                            // traits start
                            {
                                pub struct NewsArticle{
                                  pub name : String
                                }
                                pub trait OtrT{
                                    fn fun(&self)->String;
                                }
                                pub trait Summary{
                                    fn summary(&self)->String;
                                    fn summary_author(&self)->String{
                                        self.summary()
                                    }

                                }
                                impl Summary  for NewsArticle{
                                    fn summary(&self)->String{
                                        format!("{}", self.name)
                                    }

                                }
                                pub struct Tweet{
                                    pub content : String

                                }
                                impl Summary  for Tweet{
                                    fn summary(&self)->String{
                                        format!("{}", self.content)
                                    }
                                } 

                                let ar = NewsArticle{name:String::from("")};
                                let tw = Tweet{content:String::from("")};
                                ar.summary();
                                tw.summary();


                                // traits as parameters
                                pub fn notify(item :&impl Summary){
                                    item.summary();
                                }
                                notify(&ar);
                                //
                                pub fn notify1<T:Summary>(item : &T){
                                    item.summary();
                                }
                                // Specifying Multiple Trait Bounds with the + Syntax
                                pub fn notify3(item : &(impl Summary+OtrT+PartialOrd)){
                                    item.summary();
                                }
                                pub fn notify4<T: Summary+OtrT>(item : &T ){
                                    item.summary();
                                    item.fun();
                                }

                                // podemos hacer un concepto pero usamos where
                                fn some_function<T:Summary+Clone>(t:&T)->i32{
                                    0
                                }
                                fn some_function_con_where<T>(t:&T) ->i32 where T : Summary + Clone{
                                    0

                                }
                                // Returning Types that Implement Traits
                                fn returns_summarizable()->impl Summary{
                                    Tweet{content:String::from("")}
                                }

                            }//traits end
                            // lifetimes
                            {
                                // let mut x: i32 = 2;
                                // let mut y :&i32 = &x;
                                // y =  y+1;
                                // print!("\n \n {}", y);

                            }


        }

     }


     
    
}




pub mod modulo1 {
    use std::string::*;
    use std::io::*;
    enum  Pets{
        Uno, 
        Dos, 
        Tres(i32),
        Cuatro(String)
    }
    struct Post{
        name : String
    }
    impl Post{
        pub fn new(name:String)->Post{
            Post{name}
        }        
        pub fn getAuthor(&self)->Option<String>{
            if self.name.is_empty(){
                return None 
            }
            Some(self.name.clone())
        }
        pub fn register(&self, mut name:String) {
            name=String::from("")
        }
        pub fn getPets(&self, name:&str)->Pets{
            if name == "asd"{
                Pets::Dos
            }else {
                Pets::Cuatro(String::from(name))
            }
        }
    }
    fn mi_arr(s : String)-> (std::string::String, usize) {
       let x =  s.clone().len();
        (s, x)
    }
    fn mi_weacther_result()->Result<Post>{
        Ok(Post::new(String::from("")))
    }
    pub fn modulo_fn(){
        let ar :  [u32; 3]=[1,2,3];
        for _a in ar {
            
        }
        let st : &str = "";
        let mut po = &Post::new(String::from(st));
        let name = match po.getAuthor(){
            None=>String::from("none"),
            Some(name)=>name
        };
        po.register(name);
        match po.getPets("nombre de"){
            Pets::Cuatro(s)=>println!("{}", s),
            _=>print!("")
        }
        if let Pets::Cuatro(s) = po.getPets("name: &str"){
            println!("{}",s)
        }
        let v = (1..2).collect::<Vec<i32>>();
        for nu in v {}
        let mut mm = (1..3).collect::<Vec<i32>>();
        for nu  in &(1..3).collect::<Vec<i32>>() {
            
        }
        let mut t =(1,2,Pets::Cuatro(String::from("")));
        if let  Pets::Cuatro(s) = t.2{
            print!("{}",s)
        }
        let ss :  [Pets;3]=[Pets::Dos, Pets::Dos,Pets::Cuatro(String::from(""))];
        let _sss = &ss[1..2];
        if let Ok(p) =mi_weacther_result() {
            match p.getPets(&String::from("nomre")){
                Pets::Cuatro(s)=>print!(" {} ", s),
                _=>print!("")
            }
        }
        if mi_weacther_result().is_ok()   {

        }
        mi_weacther_result().expect("");
        
        fn mif() {
            // error lens hace esta mierda
            let _vv ="";
            let _bl  = String::new();
        }
        mif();

    }

    
}

pub mod error_handling_chap7{
    use std::io::*;
    // asi se maneja el tipo de error Result
    fn mierror()->Result<()>  {
        if true {
            Ok(())
        }else {
            Err(std::io::Error::from(ErrorKind::Other))
        }

        
    }
      pub     fn mainfn(){

            match mierror(){
                Ok(()) => (),
                Err(err)=> print!("{}", err)

            }
            if let Ok(()) = mierror()  {
                print!("")
            }

            let _isook = mierror().is_ok();
            let _iserr = mierror().is_err();

    ;
        }   

        pub mod structures_ch9   {
            use std::collections::*;
            struct Bound(i32, String);
            struct Ascii(Vec<i32>);
            struct Queue {
                older:Vec<char>,
                younger:Vec<char>,
            }
            impl Queue{
                pub fn push(&mut self, c : char)  {
                    self.younger.push(c );
                }
                pub fn pop(&mut self)->Option<char>{
                    std::mem::swap(&mut self.older, &mut self.younger);
                    self.younger.pop() 
                }
                pub fn new()->Queue{
                    Queue{older:vec![], younger:vec![]}
                }
                pub fn is_empty(&self )->bool{
                     self.older.is_empty()&&self.younger.is_empty()
                }
                pub fn split( self)->(Vec<char>, Vec<char>){
                    (self.older, self.younger)
                }
            }
            pub fn main_queue(){
                let mut _q =   Queue::new();
                _q.push('s');
                let mm = _q.pop();
                if let Some(a) = mm {
                    print!("{}", a );
                }
            }

            //rc
            use std::rc::Rc;
            
            
            struct Node {
                payload : String,
                chidren : Vec<Rc<Node>>
            }
            
            impl Node{
                pub fn new()->Node{
                    
                    Node{
                        payload:String::from("payload"),
                        chidren:Vec::<Rc<Node>>::new()}
                }
                 pub fn add(mut self, n:Node){
                     self.chidren.push(Rc::new( n ))
                 }
                 
            }
            
            pub fn main_node(){ 
                let mut parent = &Node::new();
                let newn = Node::new();
                

            }

            struct TreeNode<T> {
                t : T,
                l : BinTree<T>,
                r : BinTree<T>,
            }
            enum BinTree<T>{
                Empty,
                NoEmpty(Box<TreeNode<T>>)
            }
            pub fn make_tree(){
                
               
                let l =BinTree::NoEmpty(std::boxed::Box::new(TreeNode{t:1, l : BinTree::Empty, r :  BinTree::Empty}));
                let r =BinTree::NoEmpty(std::boxed::Box::new(TreeNode{t:1, l : BinTree::Empty, r :  BinTree::Empty}));
                let parent = BinTree::NoEmpty(std::boxed::Box::new(TreeNode{t:1, l , r }));
               
            }
            pub fn shared_and_mutable(){
                
                fn mi_re(v:&[i32]){
                    for i in v{
                        print!("{}",i);
                    }
                }
                type Table = std::collections::HashMap<String,String>;
                fn mi_ref_table(t:&Table){
                    let mi_r = t;
                    for _tup in mi_r{}
                }
                let table = Table::new();
                mi_ref_table(&table);

                let  mut v =   Vec::<i32>::new();
                v.push(1);
                v.push(2);
                fn mi_ch( v:&mut Vec<i32>){
                    v[0]=1;
                }
                mi_ch(&mut  v);

            }
            use std::collections::hash_map::*;
           type ReqtFn = Box< dyn Fn(&Req)->i32> ;
            struct Req{}
            struct MiTable    {
                
                vstr : String,
                hmap : HashMap<String, fn(&Req)->i32>,
               
              hmap2 :  HashMap<String, Box<dyn Fn(&Req)->i32>>
            }
            impl MiTable {
                pub fn new()->MiTable{
                    MiTable{vstr:String::from(""), hmap: HashMap::new(), hmap2:HashMap::new()}
                }
                pub fn add(&mut self, url : &str , f :fn(r:&Req)->i32) {
                    self.hmap.insert (url.to_string(), f);

                }
                pub fn add1(&mut self, url : &str , f :Box<dyn Fn(&Req)->i32>){}
                 
            }

            trait PrintView {
                fn print(&self);
            }
            struct PrintViewS {
                name:String
            }
            impl PrintView for PrintViewS {
                fn print(&self) {  
                    println!("{}",self.name)
                }
            }
            struct ControllerMain<T> where T : PrintView {
                view:Vec<T>,
                model:Vec<String>,
                paths:Vec<String>,
            }
            impl<T :PrintView> ControllerMain <T>  {
                pub fn new()->ControllerMain <T> {
                    ControllerMain {paths:vec![],view:Vec::<T>::new(), model:vec![String::from("")] }
                }
                
                pub fn resolve(&self, request_string:&String, table:&MiTable)->Option<(usize, String)>{
                    for (index, val) in self.paths.iter().enumerate() {
                        if(val == request_string){
                            return Option::Some((index, String::from("Ok")));
                        }
                    }
                    None
                }

            }
            pub fn mi_fn_1(){
                let mut mitable = MiTable::new();
                mitable.add(&String::from("/estounpath"), |req:&Req|{2});
                mitable.add1(&String::from("/estounpath1"), Box::new(|req:&Req|{1}));
                
                let controller = ControllerMain::<PrintViewS>::new();
                controller.resolve(&String::from("estounpath"), &mitable);
                
                
                let mstr =  "http://esto/43/el/path?par=val&ptp=asd ";
                //ptp=asd&bla=bla&oreo=asd
               fn split_and (mut req : String)->Option<Vec<  String > >{
                let mut resar :Vec<  String > = vec![];
                let mut jj =  req.clone();
                   let mut c : Vec<_> = jj .match_indices("&").collect::<Vec<_>>();
                    let mut it = c.iter().peekable();
                    let i0 = it.peek()?;
                    i0.0;
                    i0.1;
                    
                    let  u =  jj.get(0..i0.0)?;
                    resar.push(u.to_string());
                  //   println!("{:?}", &jj.get(0..i0.0));
                    while let Some(pair) = it.next(){
                        let (n, s) = pair;
                        if let Some( (nextp, nexts)) = it.peek(){
                          //  println!("{},{}", n, nextp);
                          // n println!("{:?}", &jj.get(*n..*nextp));
                            let r =jj.get(*n+1..*nextp)?;
                            resar.push( r.to_string()  );
                        }

                    }
                    if(resar.is_empty()){
                        None
                    }else{
                        Some( resar)
                    }
                  
                  
                   
               }
                fn split_keyvaal ( mut req : String ) -> Option<(HashMap<String, (String, String)>)>{
                    if  let Some(i) =  req.find("?"){
                        let mut res = &req.get(i+1..).expect("");
                         let mut nexsplit = res.clone();

                         
                       let  kvindex =   nexsplit.find("=")?;
                        let mut map = HashMap::new();
                        map.insert(res.to_string(),(nexsplit.get(0..kvindex)?.to_string(),nexsplit.get(kvindex+1..)?.to_string()));
                        return Some(map)
                  
                    }
                    None 
                }
                let mut keyval = mstr.clone();
                
              
                fn analize_kv(s : &str)->Option<Vec<(String,String)>>{
                    if let Some(res)  =  split_and(s.to_string()) {
                        let mut  pt : Vec<(String,String)> = Vec::new();
                        for istr in  res {
                            let u = istr.clone();
                          let y = u.find("=").expect("");
                            let substr1 = &u.get(0..y).unwrap();
                            let substr2 = &u.get(y+1..).unwrap();
                            pt.push((substr1.to_string(), substr2.to_string()));
                            
                        }

                        
                        Some(pt)
                    }else{
                        None
                    }
                }

                fn analize_full_path(s : &str) -> Option<(HashMap<String, Vec<(String, String)>>)>{
                    let mut map :HashMap<String, Vec<(String, String)>> = HashMap::new();
                   let u = s.clone();
                    let newu = u.to_string();
                   let usizesplit =   newu.find("?").unwrap();
                   let  leftstring = newu.get(0..usizesplit).unwrap();
                   
                   println!("{:?}", leftstring );
                    let kv = newu.get(usizesplit+1..).unwrap();
                    let vect  = analize_kv(kv)?;
                    if ! vect.is_empty(){
                        map.insert(leftstring.to_string(), vect);
                    }
                    println!("{:?}", kv);
                    Some(map)
               }
                let mstr =  "http://esto/43/el/path?par=val&otro1=vbla&ptp=asd ";
                println!("{:?}",  analize_full_path(mstr).unwrap());
            
              
                

               
               let mut hm : HashMap<String, Box<dyn Fn()->i32>> = HashMap::new();
               
               hm.insert("d".to_string(), Box::new(||{3}));

               let m =  hm.get_key_value("d").unwrap();
               hm.get("d").map(|f|f());
               let mfn = m.1;
               mfn();

            //    https://gist.github.com/aisamanra/da7cdde67fc3dfee00d3
                fn split_controller( mut req : String ){
                    let mut mv =  req
             
                    .trim()
                    .trim_end()
                    .trim_start()
                    .split("/").collect::<Vec<_>>();
                    let mm = mv[0];
                    println!("{}", mm  );
                    for i in  mv.iter(){
                     if let Ok(n) =  i.parse::<f32>(){
                         println!("un numero! {}", n as f32 );
                     }
                       println!("{}",i,  );
                    } 
                }
              
                type CallBack =Box< (dyn Fn()->()+'static)>;
                fn make_callback<'a,F>(f:F) ->CallBack  where F :Fn( )->() + 'static {
                    Box::new(f) as CallBack
                }
                
                  
                fn get_callback() {

                    let mut mapcalls : HashMap<String, CallBack> = HashMap::new();
                    mapcalls.insert("a".to_string(),make_callback(||{})) ;
                      let callback_ = mapcalls.get_key_value("a").unwrap().1;
                     callback_();
                }
                get_callback();
              
              
                
            }
        }


        

}
pub mod read_write_buffers {
    use std::io::*;
    use std::fs::*;
   pub enum TypeFile  {
        templates(String),
        file_standard(String)
    }
    type vec_types = Vec<TypeFile>;
    fn main_read_write_buffers_()->std::io::Result<Vec<TypeFile>> {
        
        let mut mivec : Vec<TypeFile> = Vec::new();
        
        // file open
        let dir = std::fs::read_dir(".")?;
        for entries in dir{
           let entry =  entries?;
          //println!("{:?}", entry.file_name());
           let path = entry.path();
         
          if  path.is_file()  {
            if let Some(extension) =  path.extension(){
              if extension=="txt"||extension=="html" {

                    let res : Result<String> = std::fs::read_to_string(path);
                    let thecontent :String = res.unwrap();
                   // println!("{}",  thecontent);
                    mivec.push(TypeFile::file_standard(thecontent));
              }
            }
          }
        }
          

        Ok(mivec)
    } //main_read_write_buffers_

    use std::str;

    fn search_files( s : &[u8])->std::io::Result<()>{
        if let Ok(res) = std::str::from_utf8(s){
            // el bor
            let res = String::from(res);
            
            let mut colstrin : Vec<&str> = res.split("\n").collect::<Vec<&str>>();
            
            let mut newtt : Vec<&str>= colstrin.into_iter().filter(|line|{  !line.is_empty() }). map(|line| line.trim()).collect();
            
             
            for line in newtt{
                let index:usize = match line.chars().position(|c| { c=='#'}) {
                    Some(c)=>c,
                    _ => 0 as usize
                };
            
               let nstr : Option<&str> = line.get(0..index) ;
               
               let otrstr : &str = &String::from(nstr.unwrap()) ;
               println!("{:?}",otrstr);
            }

            use std::char::*;
           let mut charsiter  =  "vlavla vlavla".chars();
           struct StartEnd (u32, u32);
           let proc :StartEnd = StartEnd(0,0);
           while let Some(c) = charsiter.next(){
             if c == '#'{
                println!("{:?}" ,c );
             }
            
           }
           
            
        //    let colstrin : Vec<&str> = String::from(res).split("\n").collect::<Vec<&str>>();
        //    colstrin.len();
        }
         
        Ok(())
    }
    fn recover_files(rest: Vec<TypeFile>){
          
        let mut  vs : Vec<TypeFile> =  rest ;
        for typefiles in & vs {
          let text :&String =  match  typefiles{
              TypeFile::file_standard(s)=> s,
              TypeFile::templates(s)=> s,
               
           };
           ;
       
        }
    }
    pub fn mi_main(){
        let mres : vec_types =  main_read_write_buffers_().unwrap();
        recover_files(mres);
        let mi:&[u8]="esto #include es la cosa\n es otra cosa \n".as_bytes();
        search_files(mi);
    }
} // read_write_buffers

pub mod rust_programing_cookbook{
    pub enum AppEnum{
        Mesn(String),
        WrapperIoErr(std::io::Error),
        Unknown
    }
     impl AppEnum{
        pub fn print_err(&self) ->std::io::Result<()> {
           let kind =  match self{
                AppEnum::Mesn(msn)=>"",
                _=>""
            };
            Ok(())

        }
     }
   
    pub fn mi_work()->std::io::Result<()>{
         Ok(())
    }

    fn main_2() {
        let mut s : Option<i32> = Some(21);
        let ss = s.take();
        assert!(s.is_none());

        let mut o = Some(12);
        let mut res12 = o.replace(1212);
        let res =  res12.take();


        fn literal_match(u:usize)->std::io::Result<String>{
            match u{
                0 | 1 => Ok(String::from("is o or 1")),
                2..=9=>Ok(String::from("inclusive range is 2 or 9")),

                _=>Ok(String::from("is ok")),
            }
        }
       // file:///G:/emule/Rust%20Programming%20Cookbook%20-%20Fast%20and%20Secure%20Apps%20(2019%20Claus%20Matzinger%3B%20Packt).pdf
        fn literal_match_tuple(u: (i32, i32, i32))->std::io::Result<String>{
             match u {
                (_, _, _)=>Ok(String::from("cool")),
                _=>Ok(String::from("cool"))
             }

        }

    }

    pub fn main() {
        
    }
    
}