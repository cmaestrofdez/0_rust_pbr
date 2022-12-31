use std::{io::{Result, BufRead}, fmt::format, fs};
use actix_web::{web,guard,HttpRequest,HttpResponse, HttpServer, App,get, cookie::Cookie, Responder, body::BoxBody, http::header::ContentType, web::ServiceConfig, HttpResponseBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{*};
use std::string::*;
#[derive(Deserialize, Debug)]
struct Info{
    userid: u32,
    username:String,
}

#[get("/users/{userid}/{username}")]
async fn index(info : actix_web::web::Path<Info>)->std::io::Result<String>{
    println!("wtf {:?}", info);
    // let (userid, username ) = info.into_inner();
    let userid = info.into_inner().userid;
    Ok(format!("dsadjlajd"))
}


#[get("/non_type_safe_alternative")]
async fn get_non_type_safe_alternative(req : HttpRequest)->std::io::Result<String>{
    let cookies = req.cookies().unwrap();
    let g = cookies.to_owned().into_iter();
    g.map(|c|{
        println!("{:?}", c);
        c
    });
    Ok(format!("format!/non_type_safe_alternative"))
}
#[get("/add_cookie")]
async fn add_cookie()->HttpResponse{
    HttpResponse::Ok()
        .cookie(Cookie::build("name", "value").finish()) 
        .cookie(Cookie::build("SameSite ", "").finish())
        .finish()
}

#[derive(Serialize)]
struct JsonRequestCustom{
    name :String
}


#[derive(Deserialize, Debug)]
struct  QueryInfo{
        name:String

}
#[get("/get_query")]
async fn add_query(query : actix_web::web::Query<QueryInfo>)-> HttpResponse {
    println!("{:?}",query.name.is_empty() );

    HttpResponse::Ok().json(JsonRequestCustom{name:String::from("daadsas")})

}




// responder
#[derive(Serialize)]
struct  MiObj{
    u:usize, 
    v :Vec<u32>
}
impl Default for MiObj{
    fn default() -> Self {
        MiObj { u : 0, v : vec![0,1,2,3,4]}
    }
}
impl Responder for MiObj {
    type Body = BoxBody;

    fn respond_to(self, req: &HttpRequest) -> HttpResponse<Self::Body> {
        ;
        HttpResponse::Ok()
            .content_type(ContentType::plaintext()) 
            .body(String::from(serde_json::to_string(&self)
            .unwrap())) 
    }
    
}

#[get("/getresponder")]
async fn get_responder() -> impl Responder{
    MiObj ::default()
}



async fn fallback_route()->String{
    String::from("fallback_route")
}
async fn handler_query(req : HttpRequest)->std::io::Result<String>{
    let connection_info     = req.  connection_info().clone();
 
    let hosta = connection_info.to_owned();
    let host  = hosta.host();
     let scheme = hosta .scheme();
    let ip =  hosta.realip_remote_addr().unwrap();
     Ok(format!("{host}, {scheme},{ip}"))
}
fn add_new_routes(cfg: &mut ServiceConfig){
    let resource = web::resource("/web/fallback").to(fallback_route);
   let resource1 =  web::resource("/otr")
                .route(web::get()).to(fallback_route)
                .route(web::post() ).to(fallback_route);
    cfg.service(resource);
    cfg.service(resource1);


    let resource3 = web::resource("/res/{name_query}")
                        .name("user_detail")
                        .route(web::get().to(handler_query));
    cfg.service(resource3);
}

// user/show
#[get("/show")]
async fn show_users(req : HttpRequest)-> HttpResponse{
    let m : MiObj = MiObj::default(); 
    HttpResponse::Ok().content_type(ContentType::json()).body( serde_json::to_string(&m).unwrap() )
     
}

// user/detail/id
#[get("/detail/{id}/{name}")]
async fn detail_user_id(path : web::Path<(u32, String)>)->Result<String>{
   let path =  path.into_inner();
   let id  = path.0;;
   let name  = path.1;
    Ok(format!("{id}, {name}"))
}


fn add_service_scope(cfg: &mut ServiceConfig){
 
    cfg.service(   web::scope("/user").service(show_users)
                .service(detail_user_id))
              .service(show_users);

}

enum TypeFile{
    NoneType,
    TxtPlain( (String, String))
}

  fn mi_search(filena : String)->Option<TypeFile>{
    let  filena = filena. clone();
    let v : Vec<&str> = filena.split('.').collect();
    if  v.len() == 2 && v[1] == "txt"{
         
        return Some(TypeFile::TxtPlain((v[0].to_string(), v[1].to_string())));
    }
    Option::None

}
fn mi_str_out()->String{
    "".to_string()
}

#[get("/{filemame}")]
async fn get_filename(path : web::Path<(String )>)-> impl Responder{
    let path= path.into_inner();
    let filename = path;
    let v : Vec<_> = fs::read_dir("./").unwrap().enumerate().map(|f|{
        f.0;
        if let  Ok(p) = f.1 {
           
            let os =  p.file_name();
            let mut filen : String= String::from(os.to_str().unwrap());
            let x =match mi_search(filen)   {
                None =>("",""),
                Some(TypeFile::TxtPlain(u))=> u,
                _ => ("","")
            };
           return  ();
            // filen = filen.trim().split("."). to_string();
            // println!("p.file_name() , {:?} ,{}",p.file_name(), filen);
            
            
        }
        return  ();
    }).collect();
println!("{:?}",v);
    format!("esto es la salida,{filename}")
}

#[actix_web::main]
async  fn main() -> std::io::Result<()>{

HttpServer::new(move||  { 
         App::new()
            . service(index)
            .service(get_non_type_safe_alternative)
            .service(add_cookie)
            .service(add_query)
            .service(get_responder)
            .service(get_filename)
            .configure(add_service_scope)
            .configure(add_new_routes)
        
     }
    )

    .bind(("127.0.0.1",3004))?
    .run()
    .await
 
}