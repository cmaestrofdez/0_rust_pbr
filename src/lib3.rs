use std::fs::OpenOptions;
use std::io::{Error, Write, Result}; 
pub mod lib3_learning{
    use std::fs::File;
    use crate::lib3::OpenOptions;
    use std::io::{Error, Write, Result}; 
    use std::io::prelude::*;
    use std::io::{ Read, BufReader , SeekFrom};
    use std::{str, vec};
    use serde::{Serialize, Deserialize};
    use bincode;
    #[derive(Serialize, Deserialize, Default, Debug)]
    struct HeaderF{
        magic:[u8;4],
        pos:i32,
        pad : [u8;2],
        pad2 : [u8;2], 
    }
    #[derive(Serialize, Deserialize, Debug)]
    struct Point {
        x : i32 
    }
pub fn lib3_learning_main()->std::io::Result<()>{
    
let mut buffer = File::create("foo.txt")?;
// buffer.write_all(b"some bytes");

          let filename : String = "mi_file.txt".to_string();
        let mut file : File = match OpenOptions::new() .read(true).write(true).append(false).create_new(false).open(filename){
            Ok(f)=>f,
            std::io::Result::Err(_)=>panic!()
        };
       let r =  std::io::Read::by_ref(&mut file);

       //  read_lines(r);
    // write_struct_fn();
    //    read_struct_fn();
    write_serialize_into();
       return Ok(());
    //    if file.metadata()?.is_file(){
    //         let mut file1 : File = file;
    //         let header_size : usize =  std::mem::size_of::<HeaderF>();
    //         let pos : u64 = file1.seek(SeekFrom::Current(0))?;
    //         let mut buffer_array_u8  =  vec![0u8; header_size as usize];
    //         let mut bufferread = BufReader::new(file1);

    //         // toma un trozo del bffer reader y lo transfiere a chunk. 
    //         // este chunk es a su vez un buffer reader
    //        let  mut chunk = bufferread.take(header_size as u64);
    //        let mut chunk_vec : Vec<u8> = vec![];
    //        chunk.read_to_end(& mut chunk_vec);
    //        let s  =  String::from_utf8_lossy(&chunk_vec)  ;  
    //      //   bufferread.read_exact(& mut buffer_array_u8)?;
    //         println!("{} ,  {:?}", header_size, s);
    //         // let mut content = String::new();
    //         // let size : usize = bufferread.capacity();
           
    //         // // let pos : u64 = file.seek(SeekFrom::Start(0))?;
    //         //  bufferread.read_to_string(&mut content)?;
    //         //  print!("size: {}, content {}", size ,  content);

    //    }
      //  writeln!( file , "hola!1 ")?;
        Ok(())

    }
    pub fn read_using_file_seek(file : File)->std::io::Result<()>{
        if file.metadata()?.is_file(){
            let mut file1 : File = file;
            let header_size : usize =  std::mem::size_of::<HeaderF>();
            let pos : u64 = file1.seek(SeekFrom::Current(0))?;
            let mut buffer_array_u8  =  vec![0u8; header_size as usize];
            let mut bufferread = BufReader::new(file1);
            bufferread.read_exact(& mut buffer_array_u8)?;
            
        }
        Ok(())

    }
    pub fn read_lines(file : &File)->std::io::Result<()>{
        let file :BufReader<&File> = BufReader::new(file);

        for (num, line ) in  file.lines().enumerate(){
            println!("num {}, line {:}", num, line? );
        }
        Ok(())
    }

    /**
     * https://adventures.michaelfbryan.com/posts/deserializing-binary-data-files/
     */
    pub fn write_struct_fn() ->std::io::Result<()>{
        let p : Point =  Point{x:2341};
        let filename : String = "mi_file.bin".to_string();
       let pseri : Vec<u8>= bincode::serialize(&p).unwrap();
       let mut file : File = match OpenOptions::new() .read(true).write(true).append(false).create_new(true).open(filename){
        Ok(f)=>f,
        std::io::Result::Err(_)=>panic!()
    };
        file.write_all(&pseri);
        println!("  {:?}--------------" , pseri );
        Ok(())
    }
    // https://stackoverflow.com/questions/56188291/how-can-i-deserialize-a-bincode-field-without-a-given-length
    // https://rust-by-example-ext.com/serde/bincode.html
    pub fn read_struct_fn()->std::io::Result<()>{
        let filename : String = "mi_file.bin".to_string();
        let mut file : File = match OpenOptions::new() .read(true).write(true).append(false).create_new(false).open(filename){
            Ok(f)=>f,
            std::io::Result::Err(_)=>panic!()
        };
       
       let datameta =  file.metadata()?;
       print!("{:?}",datameta);
        let mut file =  BufReader::new(file);
        let mut pseri : Vec<u8> = vec![];
        file.read_exact(&mut pseri);
       
        print!("{:?}", pseri);
        let bewpoint : Point = match  bincode::deserialize(&mut pseri){
             Ok(a)=>a,
             _=> Point{x:0}
        };
       
        Ok(())
    }

     
    #[derive(Serialize, Deserialize, Default, Debug)]
    pub struct Table<T>{
        v : Vec<T>,
    }
    
    
    #[derive(Serialize, Deserialize, Debug)]
    pub struct MessageSerizal{
        id : u32,
        v : Vec<u8>,
        vv : Table<u32>

    }
    impl Default for MessageSerizal {
        fn default() -> Self {
            MessageSerizal{id:0, v : vec![], vv : Table{v:vec![]}}
        }
    }

    pub fn write_serialize_into()->Result<()>{
         let mut v :Vec<MessageSerizal> = vec![MessageSerizal::default(),MessageSerizal::default(),MessageSerizal::default(),];
        let mut vector_encoded  :  Vec<u8>=  bincode::serialize(&v).unwrap();
       
       let filename : String = "mi_file.bin".to_string();
       let filename_1 = filename.clone();
      if std::fs::metadata(filename_1).is_ok(){
        std::fs::remove_file(filename.clone().clone());
        bincode::serialize_into(std::io::BufWriter::new(File::create(&filename.clone()).unwrap()), &vector_encoded);
      }
       
      if std::fs::metadata(filename.clone()).is_ok(){
        let vs : Vec<MessageSerizal> =  bincode::deserialize_from(BufReader::new(File::open(filename.clone()).unwrap())).unwrap();
        println!("{:?}", vs);
      }
    
    //    let mut file : File = OpenOptions::new() .read(true).write(true).append(false).create_new(true).open(filename).unwrap();
        // let mut vres :Vec<MessageSerizal> = bincode::deserialize(&mut vector_encoded).unwrap();
        // println!("{:?}", vres)
       
       
        //  let mut file = std::io::BufWriter::new(File::create("mi_file.bin").unwrap());
        //  bincode::serialize_into(file, &vector_encoded);
        //  match bincode::serialize_into(), &vector_encoded){
        //     Ok(a)=>a,
        //     Err(e)=>eprintln!("Application error: {e}")
        //  }
    //     // read data
    // //    let mut read_vec = vec![];
    //     // let mut fileinput = BufReader::new(File::open("data_serialize.bin").unwrap());
     
       
    //     // fileinput.read_to_end(& mut read_vec)?;
         
    //     //   let mut cursor = &read_vec[..];
    //     let noddy_vector: Vec<MessageSerizal> = bincode::deserialize_from(BufReader::new(File::open("data_serialize.bin").unwrap())).unwrap();
    //      println!("{:?}", noddy_vector);

        Ok(())
 
    }
}