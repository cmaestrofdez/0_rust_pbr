#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]

use std::{fs, path::Path, cell::{RefCell, RefMut}, rc::Rc};

use palette::Srgb;

use crate::{imagefilm::{self, ImgFilm,  FilterFilmBox, FilterFilm}, Point2f};

#[test]
pub fn test_newfilm()-> std::io::Result<()> {
    let dirpath : &Path = Path::new("filtertest");
    fs::create_dir_all(dirpath)?;
    let w = 256;
    let h  = 256;
    let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (3.0,3.0)));
    let mut film = ImgFilm::new(w,h,  filterbox, 16); 
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               film.add_sample(&Point2f::new(fx +0.5 ,fy + 0.5), Srgb::new(cf0,cf1,1.0));
        }
    }
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               film.add_sample(&Point2f::new(fx   ,fy ), Srgb::new(0.0,0.0,1.0));
        }
    }
    

    film.commit_and_write(dirpath .join("raytracer3_film_test.png").as_os_str().to_str().unwrap());



    let mut filmbox = ImgFilm::from_filter_box(w, h, (1.0, 1.0)) ;
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmbox.add_sample(&Point2f::new(fx +0.5 ,fy + 0.5), Srgb::new(cf0,cf1,1.0));
        }
    }
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmbox.add_sample(&Point2f::new(fx   ,fy ), Srgb::new(0.0,0.0,1.0));
        }
    }
 
    filmbox.commit_and_write(   dirpath .join("raytracer3_film_box_test.png").as_os_str().to_str().unwrap() );



println!("raytracer3_film_box_test.png");




    let mut filmtriangle = ImgFilm::from_filter_triangle(w, h) ;
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmtriangle.add_sample(&Point2f::new(fx +0.5 ,fy + 0.5), Srgb::new(cf0,cf1,1.0));
        }
    }
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmtriangle.add_sample(&Point2f::new(fx   ,fy ), Srgb::new(0.0,0.0,1.0));
        }
    }
   
    filmtriangle.commit_and_write( dirpath .join("raytracer3_film_triangle_test.png").as_os_str().to_str().unwrap());


    println!("raytracer3_film_triangle_test.png");









    let mut filmlanczos = ImgFilm::from_filter_lanczos(w, h) ;
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmlanczos.add_sample(&Point2f::new(fx +0.5 ,fy + 0.5), Srgb::new(cf0,cf1,1.0));
        }
    }
    for iy in 0..256{
        let fy = iy as f64;
        for ix in 0..256{
            let fx = ix as f64;
               let cf0   =  ix as f32 / 256.0;
               let cf1  =  iy as f32 / 256.0;
               filmlanczos.add_sample(&Point2f::new(fx   ,fy ), Srgb::new(0.0,0.0,1.0));
        }
    }
   
    filmlanczos.commit_and_write( dirpath .join("raytracer3_film_lanczos_test.png").as_os_str().to_str().unwrap());


    println!("raytracer3_film_lanczos_test.png");

    // multiples sizes of filter radious

    {
        let vradios = vec![(1.0,1.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (5.0, 5.0)];
        for sizes in vradios.into_iter() {
            let w = 256;
            let h  = 256;
            let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( sizes));
            let mut film = ImgFilm::new(w,h,  filterbox, 16); 
            for iy in 0..256{
                let fy = iy as f64;
                for ix in 0..256{
                    let fx = ix as f64;
                       let cf0   =  ix as f32 / 256.0;
                       let cf1  =  iy as f32 / 256.0;
                       film.add_sample(&Point2f::new(fx +0.5 ,fy + 0.5), Srgb::new(cf0,cf1,1.0));
                }
            }
            for iy in 0..256{
                let fy = iy as f64;
                for ix in 0..256{
                    let fx = ix as f64;
                       let cf0   =  ix as f32 / 256.0;
                       let cf1  =  iy as f32 / 256.0;
                       film.add_sample(&Point2f::new(fx   ,fy ), Srgb::new(0.0,0.0,1.0));
                }
            }
            let spath =  &format!("raytracer3_film_test_filter_radious{}_{}.png",sizes.0, sizes.1);
            film.commit_and_write(dirpath .join(spath).as_os_str().to_str().unwrap());
            println!("{:?}",spath );
        }
    }
    if true {
        //insert a multiples samples with variable length 
        let vradius = vec![(1.0,1.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (5.0, 5.0)];
        let samples = &vec![(1.0,1.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (5.0, 5.0)];
        for sizes in vradius.into_iter() {
            let w = 256;
            let h  = 256;
            let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( sizes));
            let mut film = ImgFilm::new(w,h,  filterbox, 16); 
            for iy in 0..256{
                let fy = iy as f64;
                for ix in 0..256{
                    let fx = ix as f64;
                       let cf0   =  ix as f32 / 256.0;
                       let cf1  =  iy as f32 / 256.0;
                       for sa  in  samples.into_iter(){ 
                        // println!("{},{},{:?}", ix, iy, sa);
                       if  ix == 253 &&  iy == 0 && sa.0 == 5.0 {
                        let kk = 1;
                       }
                         film.add_sample(&Point2f::new(fx +0.5+ sa.0 ,fy + 0.5+ sa.1), Srgb::new(cf0,cf1,1.0));
                       }
                       
                }
            }
            for iy in 0..256{
                let fy = iy as f64;
                for ix in 0..256{
                    let fx = ix as f64;
                       let cf0   =  ix as f32 / 256.0;
                       let cf1  =  iy as f32 / 256.0;
                       for sa  in  samples.into_iter(){ 
                            film.add_sample(&Point2f::new(fx +0.5+ sa.0 ,fy + 0.5+ sa.1), Srgb::new(cf0,cf1,1.0));
                      }
                }
            }
            let spath = & format!("raytracer3_film_test_filter_radious_samples_test{}_{}.png",sizes.0, sizes.1);
            film.commit_and_write(dirpath .join(spath).as_os_str().to_str().unwrap());
            println!("{}",spath );
        }
    }

    Ok(())
}




/**
 * 
 * 
 * este es pattern que requiero.
 * RefCell : permite que un borrow() puede mutar.
 * pero solo solo uno.
 * para que mutilples  borrows puedan poseer el objecto
 * se requiere que pueda ser prestado varias veces...y compartido.
 * esa es la razon de Rc<>
 * esto es un pattern llamado interior mutability.
 * https://doc.rust-lang.org/book/ch15-05-interior-mutability.html
 */
fn fn_outside_render_line( mut reffilm:   RefMut<'_, ImgFilm, >  ){
// film.add_sample(&Point2f::new(0.0,0.0), Srgb::new(1.0,1.0,1.0));
reffilm.add_sample(&Point2f::new(0.0,0.0), Srgb::new(1.0,1.0,1.0));
}

#[test]
fn test_lambda(){
   
    let reffilm =Rc::new( RefCell ::new(ImgFilm::from_filter_box(4, 4, (1.0, 1.0))));

     
   let  band =  0..4 ;
   band.for_each(|f|{
        fn_outside_render_line( reffilm.borrow_mut());
   });
     
   let refmutfilm =  reffilm.to_owned();
   let vvv = refmutfilm.borrow();
   vvv.log();
   //  ow.log();
}