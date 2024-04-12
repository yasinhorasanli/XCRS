<script lang="ts" setup>
import type { RecommendationResult } from '~/models/result';

const myResult = ref<RecommendationResult>();


</script>


<template>
    <div class=" w-full flex flex-col items-center justify-center">
        <div class="w-80" v-if="!myResult">
            <h2 class="text-2xl my-4  font-semibold">Explainable course recommendation system</h2>
            <p class="mb-4">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse nec nulla volutpat, gravida lacus
                nec, molestie lacus. Cras vel augue a elit sagittis imperdiet id non sapien.</p>
            <InputForm @submited="myResult = $event"></InputForm>
        </div>
        <div class="w-80 space-y-4" v-else>
            <h2 class="text-2xl my-4  font-semibold">My result</h2>
            <table class="table">
                <tbody>
                    <tr v-for="row in myResult.recommendations" :key="row.role">
                        <tr> 
                            <td>Role</td>
                            <td>{{ row.role }}</td>  
                        </tr> 
                        <tr>
                            <td>Explanation</td>
                            <td>{{ row.explanation }}</td> 
                        </tr>
                        <tr>    
                            <td>Courses</td>
                            <tr v-for="course in row.courses">
                                <tr>Title: {{ course.course }}</tr>
                                <tr>Url: {{ course.url }}</tr>
                                <tr>Explanation: {{ course.explanation }}</tr>
                            </tr>
                        </tr>                      
                    </tr>                    
                </tbody>
                <!-- <tr>
                    <td>Role</td>
                    <td v-text="myResult.recommendations[0].role"></td>
                </tr> -->
            </table>
            <!-- <div class="space-x-4">
                <span>Role:</span>
                <span v-text="myResult.role"></span>
            </div> -->
            <!--<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSdydUOOM0CvVoQUb6L4oCcmFxNKwbSOFzcBXK-jFFSodeBsYw/viewform?embedded=true" width="640" height="469" frameborder="0" marginheight="0" marginwidth="0">Yükleniyor…</iframe>
            -->
            <a class="underline text-blue-500" href="https://forms.gle/XxvdFguD6385K5kTA">Please fill the google form for the feedback</a>
            <UButton type="button" @click="myResult = undefined">
                Try again
            </UButton>
        </div>
    </div>
</template>