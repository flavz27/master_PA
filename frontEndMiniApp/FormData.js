$(document).ready(function(){

    $("#but_upload").click(function(){

        var fd = new FormData();
        var files = $('#file')[0].files[0];
        fd.append('file',files);
        $('#loading').show();
        // $.ajax({
        //     url: 'upload.php',
        //     type: 'post',
        //     data: fd,
        //     contentType: false,
        //     processData: false,
        //     success: function(response){
        //         if(response != 0){
        //             $("#img").attr("src",response); 
        //             $(".preview img").show(); // Display image element
        //         }else{
        //             alert('file not uploaded');
        //         }
        //     },
        // });

        $.ajax({
            url: 'http://160.98.47.94',
            type: 'post',
            data: fd,
            contentType: false,
            processData: false,
            success: function(response){
                if(response != 0){
                    console.log(response)
                    $(".measurements").show()
                    $("#loading").hide()
                    const dataResponse = JSON.parse(response)
                    console.log(dataResponse.hips)
                    $("#hips").html(dataResponse.hips);
                    $("#shoulders").html(dataResponse.shoulders);
                    const ratio = Number(dataResponse.hips) / Number(dataResponse.shoulders)
                    console.log(ratio)
                    $("#ratio").html(ratio);

                    // $("#img").attr("src",response); 
                    // $(".preview img").show(); // Display image element
                }else{
                    alert('file not uploaded');
                }
            },
        });
    });
});