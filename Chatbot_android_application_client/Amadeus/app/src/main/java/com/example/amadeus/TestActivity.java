package com.example.amadeus;

import android.media.AudioManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class TestActivity extends AppCompatActivity {
    public Button start_btn;
    public Button stop_btn;
    public VideoView videoView;
    private int videoTime = 0;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        videoView = findViewById(R.id.video_view);

        Uri uri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.divergence);
        videoView.setVideoURI(uri);

        final MediaController controller = new MediaController(this);
        videoView.setMediaController(controller);
        videoView.requestFocus();

        Log.d("Test", "Start");

        videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            // 동영상 재생준비가 완료된후 호출되는 메서드
            @Override
            public void onPrepared(MediaPlayer mp) {
                //이부분을 하지않으면, 맨처음에 VideoPlayer 에 검은화면이 나오므로, 해주셔야합니다~
                Log.d("Test", "before start");
                videoView.start();
                Log.d("Test", "after start");
                mp.setVolume(0.0f, 0.0f);

                controller.hide();
                controller.setEnabled(false);
                //mp.setAudioStreamType(AudioManager.STREAM_ALARM);
            }
        });

        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {

            public void onCompletion(MediaPlayer player) {
                //Toast.makeText(getApplicationContext(), "동영상 재생이 완료되었습니다.", Toast.LENGTH_LONG).show();
                finish();
            }
        });

    }

    private void playVideo() {
        // 비디오를 처음부터 재생할땐 0
        videoView.seekTo(videoTime);
        // 비디오 재생 시작
        videoView.start();
    }

    private void stopVideo() {
        if (videoView.isPlaying()){
            videoTime = videoView.getCurrentPosition();
            videoView.pause();
        }
    }


}
