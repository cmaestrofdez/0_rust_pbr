
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