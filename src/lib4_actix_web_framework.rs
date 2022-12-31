

pub mod  lib4_active_rs{
    
use actix_web::{HttpServer, App, Responder};
 
    #[derive (Clone)]
    pub struct Path{
        pub prefix : String
    }
    impl Path{
        pub fn define(&self, following_path :String)->String{
            self.prefix.to_owned() + &following_path

        }
    }


    pub mod login{
        use super::*;
        pub  async  fn  login()->  String {
            "login path".to_string()
        }
        pub async fn logout()->String{
            "path->logout->".to_string()
        }
    }
 
}