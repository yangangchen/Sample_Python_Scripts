function new_gen_cur_exp_region(e,t){e=e||"false";t=t||"false";var n=$("#cur_exp_main").height();$("#cur_exp_main").html('<div class="loaderBar"><img src="/images/fx_loading.gif"></div>').css("min-height",n);$("#cur_exp_"+e+"_tab").addClass("selected").unbind("click").siblings().each(function(){$(this).removeClass("selected");$(this).unbind("click").click(function(){new_gen_cur_exp_region($(this).attr("region"),"false")})});$.get("/currencies/Service/region",{region_ID:e,currency_ID:t},function(e){$("#cur_exp_main").css("min-height",20).html(e)})}