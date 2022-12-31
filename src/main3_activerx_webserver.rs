// mod lib3;
// use  lib3::lib3_learning;
use std::{env, fmt::format};

use actix_web::{get,HttpServer, App, Responder, guard, HttpResponse};
mod  lib4_actix_web_framework;


async fn greet(request : actix_web::HttpRequest)->impl Responder{
     let m : & actix_web::http::Method =  request.method() ;
     let hes = request.headers();
     use actix_web::http::header::HeaderName;
     let h: Vec<&HeaderName> = hes.iter().map(|a| a.0).collect();
     let resu :&HeaderName=  hes.iter().map(|a| a.0).collect::<Vec<&HeaderName> >().iter().find(|a| a.clone()==&"cookie").unwrap();
        actix_web::http::Method::GET == m;
    format!(" {} , {}" , resu,   hes.get(resu).unwrap().to_owned().to_str().unwrap())
}






// shared mutable

pub struct MiDataInApp{
    cnt : std::sync::Mutex<i32>
}
async fn add_index(data : actix_web::web::Data<MiDataInApp>)->String{
    let mut lkc = data.cnt.lock().unwrap();
    *lkc+=1;
    
    format!("mi index {lkc}"  )
}




pub
  fn factory_views(cfg: &mut actix_web::web::ServiceConfig){
    let path : lib4_actix_web_framework::lib4_active_rs::Path =lib4_actix_web_framework::lib4_active_rs::Path{ prefix : "/mihome".to_string() };
    cfg
    .route(&path.define("/h".to_string()), actix_web::web::get().to(greet))
    .route(&path.define("/login".to_string()), actix_web::web::get().to(lib4_actix_web_framework::lib4_active_rs::login::login))
    .route(&path.define("/logout".to_string()), actix_web::web::get().to(lib4_actix_web_framework::lib4_active_rs::login::logout))
    .route(&path.define("/mi_index".to_string()), actix_web::web::get().to(add_index));
}


async fn show_users(request : actix_web::HttpRequest)->impl Responder{
    "esto es  /users ,  / show_users".to_string()
}
// Using an Application Scope to Compose Applications
pub fn add_scopes(cfg : &mut actix_web::web::ServiceConfig){
    
    let scope  =  actix_web::web::scope("/users").service(actix_web::web::resource("/show_users").to(show_users));
    cfg.service(scope);
}

pub fn test_guatd(cfg : &mut actix_web::web::ServiceConfig){
    let all = actix_web::guard::Any( actix_web::guard::Get()).or(actix_web::guard::Post());
   cfg.service(actix_web::web::resource("/test_guatd_mi").route(actix_web::web::route().guard(all).to(||HttpResponse::Ok())));
    
}

// https://actix.rs/docs/extractors

#[get("/users/{user_id}/{friend}")]
async fn index_get(path : actix_web::web::Path<(u32, String)>)->std::io::Result<String>{
   let (user_id,friend) = path.into_inner();
   print!("-------------------");
    Ok(format!("{user_id}, {friend}"))
}

#[actix_web::main] 
async  fn main()->std::io::Result<()>{
   
   //  env::set_var("RUST_BACKTRACE", "full");
   //  lib3::lib3_learning::lib3_learning_main();
 
    // una instancia de data .
  ;
    
    
    HttpServer::new(move ||{
        let app = App::new()
            .configure(factory_views)
            .configure(add_scopes)
            .configure(test_guatd)
            .service(index_get)
            .app_data(  actix_web::web::Data::new(MiDataInApp{cnt:std::sync::Mutex::new(0)}).clone());
            
     
       return  app
        
    })
    .bind("127.0.0.1:4000")?
    .workers(3)
    .run().await

// Ok(())
}
